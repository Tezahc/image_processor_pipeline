import random
import math
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Tuple, List, Any, Union 
from PIL import Image, UnidentifiedImageError
from ultralytics.utils.ops import xyxy2xywhn
from deprecated import deprecated
from icecream import ic


def _convert_to_yolo_bbox(img_width: int, img_height: int, box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Les dimensions de l'image ({img_width}x{img_height}) doivent être positives.")
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = (box[0] + box[2]) / 2.0 * dw
    y_center = (box[1] + box[3]) / 2.0 * dh
    width = (box[2] - box[0]) * dw
    height = (box[3] - box[1]) * dh
    return x_center, y_center, width, height

def paste_overlay_onto_background(
    overlay_path: Path,
    background_path: Path,
    output_dirs: List[Path],

    # Reçoit les **options définies pour l'étape
    yolo_class_id: int = 0,
    min_scale: float = 0.1,
    max_scale: float = 0.5,
    **options: Any # Accepter d'autres options non utilisées
) -> Optional[List[Path]]: # Retourne une liste de 2 Path (image, label) ou None
    """
    Superpose une image overlay sur un fond en contrôlant la surface relative
    de l'overlay. Sauvegarde l'image résultante et le fichier label YOLO.

    L'overlay est redimensionné pour occuper une surface (en pixels) correspondant
    à un pourcentage aléatoire (entre min_scale et max_scale)
    de la surface totale de l'image de fond, tout en conservant ses proportions.
    L'overlay est ensuite placé aléatoirement sur le fond.

    Args:
        overlay_path (Path): Chemin vers l'image overlay (avec canal alpha).
        background_path (Path): Chemin vers l'image de fond.
        output_paths (List[Path]): Liste des chemins des dossiers de sortie.
                                   Attend au moins 2: [0] pour images, [1] pour labels.
        yolo_class_id (int): ID de classe pour le label YOLO.
        min_scale (float): Ratio minimal de la surface de l'overlay par rapport
                                   à la surface du fond (ex: 0.01 pour 1%).
        max_scale (float): Ratio maximal de la surface de l'overlay (ex: 0.05 pour 5%).
        **options (Any): Accepte d'autres options.

    Returns:
        Optional[List[Path]]:
            - List[Path]: Liste contenant [chemin_image_sauvegardée, chemin_label_sauvegardé].
            - None: Si erreur (lecture, dossiers sortie, placement impossible, sauvegarde échouée).
    """
    # --- 1. Vérifications Préliminaires ---
    if len(output_dirs) < 2:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: "
              f"Au moins 2 dossiers de sortie sont requis (images, labels), {len(output_dirs)} fourni(s).")
        return None
    image_target_dir = output_dirs[0]
    label_target_dir = output_dirs[1]

    # --- 2. Charger overlay et background ---
    try:
        overlay = Image.open(overlay_path)
        # overlay.verify()
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')
        
        background = Image.open(background_path).convert('RGB') # Assurer RGB pour sortie JPG/JPEG
        # background.verify()
        
    except FileNotFoundError as e:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Fichier non trouvé: {e}")
        return None
    except UnidentifiedImageError as e:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Impossible d'ouvrir l'image {e}")
        return None
    except TypeError as te:
        print(f"Erreur [{overlay_path.name} + {background_path.name}]: Type d'image invalide : {te}")
        return None
    except Exception as e:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Échec lecture fichiers: {e}")
        return None

    # --- 3. Calculer taille et position (avec tentatives) ---
    try:
        # récupère la diagonale du fond et défini la diagonale de l'overlay désiré
        bg_diag = math.hypot(background.width, background.height)
        target_ratio = random.uniform(min_scale, max_scale)
        ov_diag_target = bg_diag * target_ratio

        # Récupère le ratio de l'image (largeur/hauteur)
        ov_aspect_ratio = overlay.width / overlay.height

        # limite la hauteur max de l'overlay et de la diagonale correspondante
        # (on pourrait partir de la largeur, c'est symétrique)
        h_max = min(background.width / ov_aspect_ratio, background.height)
        max_ov_diag = math.hypot(ov_aspect_ratio * h_max, h_max)
        ov_diag = min(ov_diag_target, max_ov_diag)

        # Définit les dimensions finales de l'overlay
        new_ov_height = int(math.sqrt(ov_diag**2 / (ov_aspect_ratio**2 + 1)))  # tkt les maths
        new_ov_width = int(ov_aspect_ratio * new_ov_height)

        # Vérifier si l'overlay redimensionné peut tenir dans le background
        # NeverTM ? :pray:
        if new_ov_width > background.width or new_ov_height > background.height:
            print(f"Avertissement [{overlay_path.name} - OverlayPair]: Overlay redimensionné ({new_ov_width}x{new_ov_height}) "
                  f"pour ratio de surface {target_ratio:.3f} est trop grand pour le fond ({background.width}x{background.height}). "
                  "Opération annulée pour cette paire.")
            return None
        
        # Redimensionner l'overlay
        overlay_resized = overlay.resize((new_ov_width, new_ov_height), Image.Resampling.LANCZOS)
        
        # --- 4. Placer l'overlay aléatoirement ---
        # Position du coin supérieur gauche de l'overlay
        pos_x = random.randint(0, background.width - new_ov_width)
        pos_y = random.randint(0, background.height - new_ov_height)

        # --- 5. Superposition et Calcul de la BBox pour YOLO ---
        # Créer une copie du fond pour coller dessus
        composite_image = background.copy()
        composite_image.paste(overlay_resized, (pos_x, pos_y), overlay_resized)

        # Coordonnées de la BBox de l'overlay sur l'image composite (format xyxy absolu)
        # (x_min, y_min, x_max, y_max)
        bbox_xyxy_abs = np.array([pos_x, pos_y, pos_x + new_ov_width, pos_y + new_ov_height])

        # Convertir xyxy (absolu) en xywhn (YOLO normalisé)
        bbox_yolo_xywhn = xyxy2xywhn(bbox_xyxy_abs, *background.size, clip=True, eps=1e-3)
        cx, cy, w_norm, h_norm = bbox_yolo_xywhn # Extraire la première (et unique) bbox

        yolo_label_str = f"{yolo_class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"

    except ValueError as ve: # Erreurs de calcul, dimensions
        print(f"Erreur de valeur [{overlay_path.name} + {background_path.name} - OverlayPair]: {ve}")
        return None
    except Exception as e_process:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: "
              f"Échec pendant le processus de superposition: {e_process}")
        import traceback
        traceback.print_exc() # Pour debug plus détaillé

        return None

    # --- 5. Vérifier si le placement a réussi ---
    if composite_image is None or yolo_label_str is None:
        print(f"Avertissement [OverlayPair]: Impossible de placer {overlay_path.name} sur {background_path.name}.")
        return None

    # --- 6. Sauvegarde de l'image et du label ---
    saved_paths: List[Path] = []

    # Nom basé sur l'overlay, avec préfixe
    img_output_path = image_target_dir / f"{overlay_path.stem}{background_path.suffix}"
    label_output_path = label_target_dir / f"{overlay_path.stem}.txt"
    
    try:
        # Sauvegarder l'image
        composite_image.save(img_output_path)
        saved_paths.append(img_output_path)
        
        # Sauvegarder le label
        with open(label_output_path, 'w', encoding='utf-8') as f:
            f.write(yolo_label_str)
        saved_paths.append(label_output_path)

        # print(f"Info [{overlay_path.name} + {background_path.name} - OverlayPair]: Image et Label sauvegardés.")
        # --- 7. Retourner la liste des DEUX chemins ---
        return saved_paths

    except Exception as e_save:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Échec lors de la sauvegarde: {e_save}")
        # import traceback
        # traceback.print_exc()
        # Nettoyer les fichiers potentiellement créés partiellement ? Optionnel.
        # Si l'image a été sauvée mais le label échoue, saved_paths contiendra 1 élément.
        # Si on veut être strict, on supprime le fichier image aussi.
        for p in saved_paths:
            try:
                if p.exists(): p.unlink()
            except OSError:
                print(f"Avertissement: Impossible de nettoyer le fichier partiellement créé {p}")
        return None # Échec global si la sauvegarde d'un des deux échoue


@deprecated(reason="utiliser `paste_overlay_onto_background` à la place.")
def process_overlay_pair(
    overlay_path: Path,
    background_path: Path,
    output_dirs: List[Path],

    # Reçoit les **options définies pour l'étape
    yolo_class_id: int = 0,
    min_scale: float = 0.1,
    max_scale: float = 0.35,
    max_placement_attempts: int = 10, # Nombre max d'essais pour placer l'overlay
    **options: Any # Accepter d'autres options non utilisées
) -> Optional[List[Path]]: # Retourne une liste de 2 Path (image, label) ou None
    """
    Superpose une image overlay sur un fond, SAUVEGARDE l'image résultante
    et le fichier label YOLO correspondant.

    Args:
        overlay_path (Path): Chemin vers l'image overlay (avec canal alpha).
        background_path (Path): Chemin vers l'image de fond.
        output_paths (List[Path]): Liste des chemins des dossiers de sortie.
                                   Attend au moins 2: [0] pour images, [1] pour labels.
        yolo_class_id (int): ID de classe pour le label YOLO.
        min_scale (float): Échelle minimale de l'overlay.
        max_scale (float): Échelle maximale de l'overlay.
        output_prefix (str): Préfixe pour les noms des fichiers de sortie.
        output_format_image (str): Format de sauvegarde PIL pour l'image (ex: JPEG, PNG).
        max_placement_attempts (int): Nombre maximum de tentatives pour redimensionner/placer.
        **options (Any): Accepte d'autres options.

    Returns:
        Optional[List[Path]]:
            - List[Path]: Liste contenant [chemin_image_sauvegardée, chemin_label_sauvegardé].
            - None: Si erreur (lecture, dossiers sortie insuffisants, placement impossible, sauvegarde échouée).
    """
    # --- 1. Vérifications Préliminaires ---
    if len(output_dirs) < 2:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: "
              f"Au moins 2 dossiers de sortie sont requis (images, labels), {len(output_dirs)} fourni(s).")
        return None
    image_target_dir = output_dirs[0]
    label_target_dir = output_dirs[1]

    # --- 2. Charger overlay et background ---
    try:
        overlay = Image.open(overlay_path)
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')
        background = Image.open(background_path).convert('RGB') # Assurer RGB pour sortie JPG/JPEG

    except FileNotFoundError as e:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Fichier non trouvé: {e}")
        return None
    except UnidentifiedImageError as e:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Impossible d'ouvrir l'image {e}")
        return None
    except Exception as e:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Échec lecture fichiers: {e}")
        return None

    # --- 3. Calculer taille et position (avec tentatives) ---
    bg_width, bg_height = background.size
    if bg_width <= 0 or bg_height <= 0:
        print(f"Avertissement [OverlayPair]: Image de fond invalide {background_path.name} ({bg_width}x{bg_height}). Ignoré.")
        return None

    composite_image: Optional[Image.Image] = None
    yolo_label_str: Optional[str] = None
    final_bbox: Optional[Tuple[int, int, int, int]] = None # Pour debug potentiellement

    for attempt in range(max_placement_attempts):
        scale = random.uniform(min_scale, max_scale)
        base_size = min(bg_width, bg_height) * scale
        ov_width, ov_height = overlay.size
        if ov_width <=0 or ov_height <=0: # Vérifier overlay valide
             print(f"Erreur [{overlay_path.name} - OverlayPair]: Overlay a des dimensions invalides {ov_width}x{ov_height}.")
             return None # Erreur fatale pour cette paire

        # Calcul ratio basé sur dimension overlay
        if ov_width >= ov_height:
            new_width = int(base_size)
            ratio = new_width / ov_width
            new_height = int(ov_height * ratio)
        else:
            new_height = int(base_size)
            ratio = new_height / ov_height
            new_width = int(ov_width * ratio)

        if new_width <= 0 or new_height <= 0:
            continue # Essayer une autre échelle

        max_x = bg_width - new_width
        max_y = bg_height - new_height

        if max_x >= 0 and max_y >= 0:
            pos_x = random.randint(0, max_x)
            pos_y = random.randint(0, max_y)

            try:
                # Créer une COPIE du background pour cette tentative pour ne pas le modifier
                # si plusieurs tentatives sont nécessaires (même si la boucle sort au succès)
                current_background_copy = background.copy()
                overlay_resized = overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)
                # Coller sur la copie
                current_background_copy.paste(overlay_resized, (pos_x, pos_y), overlay_resized)

                # Calculer BBox et Label YOLO
                bbox = (pos_x, pos_y, pos_x + new_width, pos_y + new_height)
                yolo_coords = _convert_to_yolo_bbox(bg_width, bg_height, bbox)
                label_str = f"{yolo_class_id} {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}"

                # Succès ! Stocker les résultats et sortir de la boucle d'essais
                composite_image = current_background_copy
                yolo_label_str = label_str
                final_bbox = bbox # Garder la bbox pour info si besoin
                break # Sortir de la boucle while/for attempts

            except ValueError as e_yolo: # Erreur spécifique de _convert_to_yolo_bbox
                 print(f"Erreur [{overlay_path.name} - OverlayPair]: Erreur conversion YOLO: {e_yolo}")
                 # C'est une erreur fatale pour cette paire, on ne peut pas générer de label
                 return None
            except Exception as e_paste:
                 # Erreur pendant resize ou paste
                 print(f"Erreur [{overlay_path.name} - OverlayPair]: Échec redim/collage tentative {attempt+1}: {e_paste}")
                 # Essayer à nouveau si possible

    # --- 4. Vérifier si le placement a réussi ---
    if composite_image is None or yolo_label_str is None:
        print(f"Avertissement [OverlayPair]: Impossible de placer {overlay_path.name} sur {background_path.name} après {max_placement_attempts} tentatives.")
        return None

    # --- 5. Sauvegarde de l'image et du label ---
    saved_paths: List[Path] = []
    # Nom basé sur l'overlay, avec préfixe

    img_output_path = image_target_dir / f"{overlay_path.stem}{background_path.suffix}"
    label_output_path = label_target_dir / f"{overlay_path.stem}.txt"
    
    try:
        # Sauvegarder l'image
        composite_image.save(img_output_path)
        saved_paths.append(img_output_path)
        
        # Sauvegarder le label
        with open(label_output_path, 'w', encoding='utf-8') as f:
            f.write(yolo_label_str)
        saved_paths.append(label_output_path)

        # print(f"Info [{overlay_path.name} + {background_path.name} - OverlayPair]: Image et Label sauvegardés.")
        # --- 6. Retourner la liste des DEUX chemins ---
        return saved_paths

    except Exception as e_save:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Échec lors de la sauvegarde: {e_save}")
        import traceback
        traceback.print_exc()
        # Nettoyer les fichiers potentiellement créés partiellement ? Optionnel.
        # Si l'image a été sauvée mais le label échoue, saved_paths contiendra 1 élément.
        # Si on veut être strict, on supprime le fichier image aussi.
        for p in saved_paths:
            try:
                if p.exists(): p.unlink()
            except OSError:
                print(f"Avertissement: Impossible de nettoyer le fichier partiellement créé {p}")
        return None # Échec global si la sauvegarde d'un des deux échoue