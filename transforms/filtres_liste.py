import cv2
import numpy as np
from pathlib import Path
from typing import Any, List, Tuple, Optional
from image_processor_pipeline.utils.utils import _validate_dirs 


def process_images_with_color_masks(
    image_path: Path,
    output_dirs: List[Path],
    color_ranges_to_exclude_hsv: List[Tuple[int, int, int, int, int, int]],
    zones: List[Tuple[int, int, int, int] | None] | None = None,
    output_prefix: str = "",
    **options: Any
) -> Optional[Path]:
    """
    Traite les images d'un dossier pour rendre transparentes les zones correspondant
    à une liste de plages de couleurs HSV spécifiées.

    Args:
        input_folder: Chemin du dossier contenant les images source.
        output_folder: Chemin du dossier où sauvegarder les images traitées (sera créé s'il n'existe pas).
        color_ranges_to_exclude_hsv: Une liste de tuples. Chaque tuple contient 6 entiers
            représentant les bornes HSV d'une couleur à exclure :
            (min_H, min_S, min_V, max_H, max_S, max_V).
        zones: Zones de l'image où appliquer les filtres colorimétriques. Correspond à des marges de crop **(top, bottom, left, right)**.  
            Doit être de même dimension que `color_ranges_to_exclude_hsv`.  
            si une zone est définie comme `None`, le filtre est appliqué sur la totalité de l'image.
        output_prefix: Un préfixe optionnel à ajouter au nom de chaque fichier de sortie.
                       Par exemple, "prefix_".
    """
    # --- 1. Vérification des arguments et Setup ---
    output_dir = _validate_dirs(output_dirs, nb_dirs=1)

    if not color_ranges_to_exclude_hsv:
        raise ValueError(f"Erreur [{image_path.name} - ColorMask] : `color_ranges_to_exclude_hsv` est requis pour traiter les données")
    
    # S'assure que `zones` est de même longueur que les filtres
    if zones and len(zones) != len(color_ranges_to_exclude_hsv):
        raise ValueError(f"Les zones d'application des filtres colorimétriques ({len(zones)}) ne correspondent pas aux filtres ({len(color_ranges_to_exclude_hsv)}). Les 2 paramètres doivent être de même longueur !.")
    elif not zones:
        zones = [None] * len(color_ranges_to_exclude_hsv)
    
    # --- 2. Lecture de l'image ---
    image = cv2.imread(str(image_path))
    if image is None:
        raise IOError("Impossible de charger l'image.")

    # --- 3. Traitement des masques de couleur ---
    # Convertir en espace HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Initialiser un masque combiné vide (tout noir) de la bonne taille
    # Il accumulera toutes les zones à *exclure*
    combined_mask_to_exclude = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)

    # Boucler sur chaque plage de couleur à exclure
    for filter, zone in zip(color_ranges_to_exclude_hsv, zones):
        h_min, s_min, v_min, h_max, s_max, v_max = filter
        z_top, z_bottom, z_left, z_right = zone if zone else (0, 0, 0, 0)

        # crée un masque sur la zone définie
        zone_mask = np.zeros_like(hsv[...,0], dtype=np.uint8)
        zone_mask[z_top : hsv.shape[0] - z_bottom, z_left : hsv.shape[1] - z_right] = 255

        # Définir les bornes numpy pour la plage actuelle
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

        # Créer le masque pour cette plage de couleur spécifique
        current_color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        # L'applique uniquement sur la zone prescrite
        full_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        current_color_mask = cv2.bitwise_and(full_mask, zone_mask)

        # Ajouter (OU logique) ce masque au masque combiné
        # Les pixels correspondant à *n'importe laquelle* des plages seront blancs
        combined_mask_to_exclude = cv2.bitwise_or(combined_mask_to_exclude, current_color_mask)

    # Inverser le masque combiné :
    # Les zones à exclure (blanches dans combined_mask) deviennent noires.
    # Les zones à conserver (noires dans combined_mask) deviennent blanches.
    # Ce masque inversé `mask_inv` représente les zones à garder (opacité = 255).
    mask_to_keep = cv2.bitwise_not(combined_mask_to_exclude)

    # Appliquer le masque inversé à l'image originale pour mettre à zéro les pixels exclus
    # (bitwise_and ne conserve que les pixels où le masque est blanc)
    # result = cv2.bitwise_and(image, image, mask=mask_to_keep)

    # Préparer l'image de sortie avec canal alpha
    # Les canaux BGR viennent de 'result' (où les zones exclues sont noires)
    # Le canal Alpha vient directement de 'mask_inv' (blanc = opaque, noir = transparent)
    b, g, r = cv2.split(image)
    alpha = mask_to_keep
    result_with_alpha = cv2.merge((b, g, r, alpha))

    # --- 4. Sauvegarde de l'image résultante ---
    # Construire le chemin de sortie et sauvegarder en PNG (pour la transparence)
    output_filename = f"{output_prefix}{"_" if output_prefix else ""}{image_path.stem}.png"
    output_path = output_dir / output_filename

    try:
        success = cv2.imwrite(output_path, result_with_alpha)
        if success:
            return output_path
        else: 
            raise RuntimeError(f"Échec de sauvegarde (imwrite a retrouné False) pour {output_filename}")
    except Exception as e_save:
        print(f"Erreur lors de la sauvegarde de {output_path}: {e_save}")
        return None

# --- Exemples d'utilisation ---
if __name__ == '__main__':
    # Définir la liste des plages de couleurs HSV à exclure
    # Chaque tuple = (H_min, S_min, V_min, H_max, S_max, V_max)
    colors_to_remove = [
        # 🌕 Jaune vif / doré
        # (20, 100, 180, 40, 255, 255),
        # (0, 100, 100, 20, 255, 255),
        # (160, 100, 100, 180, 255, 255),
        # 🟡 Jaune/orangé plus doux (brun clair / caramel)
        # (15, 90, 130, 35, 220, 240),
        # 🟤 Brun olive foncé à clair (#766641, #73623f)
        # (10, 60, 60, 35, 160, 160),
        # 🟢 Vert clair (mint, plastique clair)
        # (45, 20, 160, 75, 120, 255),
        # (40, 50, 50, 80, 255, 255),
        # 🟩 Vert plus foncé ou désaturé (contours, bordure floue)
        # (40, 30, 120, 70, 180, 200),
        # (60, 20, 100, 90, 80, 160), # low
        # (60, 20, 100, 90, 80, 160), # medium
        # (55, 10, 90, 95, 100, 180), # large
        # ⚫ Bords noirs ou très sombres à faible saturation
        # (0, 0, 0, 20, 60, 90),
        # (0, 0, 0, 180, 50, 100),
        # 🟫 Bords de transition brun/noir (comme #a49756)
        # (15, 40, 80, 30, 140, 180),
        # (8, 137, 24, 28, 255, 144),
        # (7, 26, 0, 27, 146, 116),
        # ⚪ blanc / clairs
        # (0, 0, 200, 180, 50, 255),
        # 🔘 gris clair (#929292)
        # (0, 0, 130, 180, 20, 160)
        # (0, 0, 110, 180, 30, 170)
        # (0, 0, 100, 180, 40, 180),
        # (0, 0, 160, 180, 30, 210)
        (0, 0, 0, 180, 255, 150),
        # (15, 70, 200, 35, 210, 255),
        (15, 60, 200, 35, 255, 255),
        (15, 30*2.55, 55*2.55, 30, 60*2.55, 80*2.55),
        (15, 60*2.55, 60*2.55, 30, 75*2.55, 90*2.55)
    ]

    # [
    #     (20, 100, 180, 40, 255, 255),  # Jaune vif / doré
    #     (52, 20, 205, 72, 120, 255),   # Vert clair
    #     (0, 0, 0, 180, 50, 100),         # Noir ou très sombre
    #     (18, 110, 110, 38, 210, 210),  # Brun/olive clair à moyen
    #     (30, 90, 130, 50, 190, 230),   # Jaune/brun doux
    #     (41, 30, 190, 61, 130, 255)    # Vert d'eau plus foncé
    # ]

    # [
    #     (0, 0, 0, 180, 50, 100),     # couleur 1 (noir/gris peu saturé)
    #     (20, 135, 190, 32, 255, 255), # couleur 2
    #     (52, 0, 222, 65, 55, 255),    # couleur 3
    #     (24, 100, 140, 35, 160, 255), # 
    #     (15, 70, 60, 35, 160, 140), # brun orangé
    #     (22, 150, 200, 30, 255, 255) # jaune foncé
    #     # Ajoutez d'autres tuples ici si nécessaire
    # ]
    process_images_with_color_masks(
        Path("C:/Users/GuillaumeChazet/Documents/ICUREsearch/PiccMid/Training/dataset/H1_4/Videos/DSC_0067/0-raw/H1_4-frame_0032.jpg"),
        [Path("C:/Users/GuillaumeChazet/Documents/ICUREsearch/PiccMid/Training/dataset/H1_4/Videos/DSC_0067")],
        color_ranges_to_exclude_hsv=colors_to_remove,
        zones=[
            (0,0,0,0),
            (0,0,0,0),
            (0,0,0,0),
            (0,0,0,0)
        ]
    )