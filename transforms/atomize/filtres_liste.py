import cv2
import numpy as np
import os
from typing import List, Tuple, Optional # Pour les annotations de type (optionnel mais recommandé)
from tqdm.notebook import tqdm

def process_images_with_color_masks(
    input_folder: str,
    output_folder: str,
    color_ranges_to_exclude_hsv: List[Tuple[int, int, int, int, int, int]],
    output_prefix: str = ""
):
    """
    Traite les images d'un dossier pour rendre transparentes les zones correspondant
    à une liste de plages de couleurs HSV spécifiées.

    Args:
        input_folder: Chemin du dossier contenant les images source.
        output_folder: Chemin du dossier où sauvegarder les images traitées (sera créé s'il n'existe pas).
        color_ranges_to_exclude_hsv: Une liste de tuples. Chaque tuple contient 6 entiers
            représentant les bornes HSV d'une couleur à exclure :
            (min_H, min_S, min_V, max_H, max_S, max_V).
        output_prefix: Un préfixe optionnel à ajouter au nom de chaque fichier de sortie.
                       Par exemple, "prefix_".
    """
    if not os.path.isdir(input_folder):
        print(f"Erreur : Le dossier d'entrée n'existe pas : {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Dossier d'entrée : {input_folder}")
    print(f"Dossier de sortie : {output_folder}")
    print(f"Préfixe de sortie : '{output_prefix}'")
    # print(f"Plages de couleurs HSV à exclure : {color_ranges_to_exclude_hsv}")

    processed_count = 0
    error_count = 0

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print("Aucun fichier image trouvé dans le dossier d'entrée.")
        return

    print(f"Traitement de {len(image_files)} images...")

    for image_name in tqdm(image_files):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Erreur : Impossible de charger l'image {image_name}. Passage à la suivante.")
            error_count += 1
            continue

        # Convertir en espace HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Initialiser un masque combiné vide (tout noir) de la bonne taille
        # Il accumulera toutes les zones à *exclure*
        combined_mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)

        # Boucler sur chaque plage de couleur à exclure
        for h_min, s_min, v_min, h_max, s_max, v_max in color_ranges_to_exclude_hsv:
            # Définir les bornes numpy pour la plage actuelle
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])

            # Créer le masque pour cette plage de couleur spécifique
            current_mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # Ajouter (OU logique) ce masque au masque combiné
            # Les pixels correspondant à *n'importe laquelle* des plages seront blancs
            combined_mask = cv2.bitwise_or(combined_mask, current_mask)

        # Inverser le masque combiné :
        # Les zones à exclure (blanches dans combined_mask) deviennent noires.
        # Les zones à conserver (noires dans combined_mask) deviennent blanches.
        # Ce masque inversé `mask_inv` représente les zones à garder (opacité = 255).
        mask_inv = cv2.bitwise_not(combined_mask)

        # Appliquer le masque inversé à l'image originale pour mettre à zéro les pixels exclus
        # (bitwise_and ne conserve que les pixels où le masque est blanc)
        result = cv2.bitwise_and(image, image, mask=mask_inv)

        # Préparer l'image de sortie avec canal alpha
        # Les canaux BGR viennent de 'result' (où les zones exclues sont noires)
        # Le canal Alpha vient directement de 'mask_inv' (blanc = opaque, noir = transparent)
        b, g, r = cv2.split(result)
        alpha = mask_inv
        result_with_alpha = cv2.merge((b, g, r, alpha))

        # Construire le chemin de sortie et sauvegarder en PNG (pour la transparence)
        base_name, _ = os.path.splitext(image_name)
        output_filename = f"{output_prefix}_{base_name}.png"
        output_path = os.path.join(output_folder, output_filename)

        try:
            cv2.imwrite(output_path, result_with_alpha)
            # print(f"Image sauvegardée : {output_path}") # Décommenter si besoin de voir chaque fichier
            processed_count += 1
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de {output_path}: {e}")
            error_count += 1

    print("-" * 30)
    print("Traitement terminé.")
    print(f"{processed_count} images traitées avec succès.")
    if error_count > 0:
        print(f"{error_count} erreurs rencontrées.")
    print(f"Les images filtrées sont enregistrées dans : {output_folder}")

# --- Exemple d'utilisation ---
if __name__ == '__main__':
    # Définir les chemins (adaptez-les à votre configuration)
    input_dir = r"C:\Users\GuillaumeChazet\Documents\ICUREsearch\PiccMid\Training\dataset\H1_2\Videos\DSC_0059\1-crop"
    output_dir = r"C:\Users\GuillaumeChazet\Documents\ICUREsearch\PiccMid\Training\dataset\H1_2\Videos\DSC_0059\2-filter_large"

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
        (0, 0, 100, 180, 40, 180)
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

    # Définir le préfixe pour les fichiers de sortie
    output_file_prefix = "H1_2"

    # Appeler la fonction
    process_images_with_color_masks(input_dir, output_dir, colors_to_remove, output_file_prefix)