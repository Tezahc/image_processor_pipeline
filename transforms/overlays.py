import random
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any, Union 
from PIL import Image, UnidentifiedImageError


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

def process_overlay_pair(
    overlay_path: Path,
    background_path: Path,
    output_paths: List[Path],

    # Reçoit les **options définies pour l'étape
    yolo_class_id: int = 0,
    min_scale: float = 0.1,
    max_scale: float = 0.35,
    output_format_image: str = "jpg", # Format pour l'image composite
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
    if len(output_paths) < 2:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: "
              f"Au moins 2 dossiers de sortie sont requis (images, labels), {len(output_paths)} fourni(s).")
        return None
    image_target_dir = output_paths[0]
    label_target_dir = output_paths[1]

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
    out_img_suffix = f".{output_format_image.lower()}"
    if output_format_image.lower() == "jpeg": out_img_suffix = ".jpg"

    img_output_path = image_target_dir / f"{overlay_path.stem}{out_img_suffix}"
    label_output_path = label_target_dir / f"{overlay_path.stem}.txt"

    try:
        # Sauvegarder l'image
        composite_image.save(img_output_path, format=output_format_image)
        saved_paths.append(img_output_path)

        # Sauvegarder le label
        with open(label_output_path, 'w', encoding='utf-8') as f:
            f.write(yolo_label_str)
        saved_paths.append(label_output_path)

        print(f"Info [{overlay_path.name} + {background_path.name} - OverlayPair]: Image et Label sauvegardés.")
        # --- 6. Retourner la liste des DEUX chemins ---
        return saved_paths

    except Exception as e_save:
        print(f"Erreur [{overlay_path.name} + {background_path.name} - OverlayPair]: Échec lors de la sauvegarde: {e_save}")
        # Nettoyer les fichiers potentiellement créés partiellement ? Optionnel.
        # Si l'image a été sauvée mais le label échoue, saved_paths contiendra 1 élément.
        # Si on veut être strict, on supprime le fichier image aussi.
        for p in saved_paths:
            try:
                if p.exists(): p.unlink()
            except OSError:
                print(f"Avertissement: Impossible de nettoyer le fichier partiellement créé {p}")
        return None # Échec global si la sauvegarde d'un des deux échoue