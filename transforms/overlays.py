import random
from pathlib import Path
from typing import Optional, Dict, Tuple
from PIL import Image, UnidentifiedImageError

# Le helper _convert_to_yolo_bbox est toujours utile
def _convert_to_yolo_bbox(img_width: int, img_height: int, box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    # ... (même code que précédemment) ...
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
    yolo_class_id: int = 0,
    min_scale: float = 0.1,
    max_scale: float = 0.35,
) -> Optional[Dict[str, object]]:
    """
    Superpose une image overlay sur une image de fond spécifique.

    Prend les deux chemins en entrée, calcule une taille/position aléatoire,
    effectue la superposition et retourne l'image résultante et le label YOLO.

    Args:
        overlay_path (Path): Chemin vers l'image overlay (doit avoir un canal alpha).
        background_path (Path): Chemin vers l'image de fond.
        yolo_class_id (int): L'ID de classe à utiliser dans le fichier label YOLO.
        min_scale (float): Échelle minimale de l'overlay.
        max_scale (float): Échelle maximale de l'overlay.

    Returns:
        Optional[Dict[str, object]]: Dictionnaire contenant :
            {'image': Image.Image, 'label': str}
            Retourne None si une erreur se produit.
    """
    try:
        # --- 1. Charger overlay et background ---
        overlay = Image.open(overlay_path)
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')

        background = Image.open(background_path).convert('RGB')

        # --- 2. Calculer taille et position ---
        bg_width, bg_height = background.size
        if bg_width <= 0 or bg_height <= 0:
            print(f"Avertissement [OverlayPair]: Image de fond invalide {background_path.name} ({bg_width}x{bg_height}). Ignoré.")
            return None

        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            attempts += 1
            scale = random.uniform(min_scale, max_scale)
            base_size = min(bg_width, bg_height) * scale

            ov_width, ov_height = overlay.size
            if ov_width >= ov_height:
                new_width = int(base_size)
                ratio = new_width / ov_width if ov_width > 0 else 0
                new_height = int(ov_height * ratio)
            else:
                new_height = int(base_size)
                ratio = new_height / ov_height if ov_height > 0 else 0
                new_width = int(ov_width * ratio)

            if new_width <= 0 or new_height <= 0:
                continue

            max_x = bg_width - new_width
            max_y = bg_height - new_height

            if max_x >= 0 and max_y >= 0:
                pos_x = random.randint(0, max_x)
                pos_y = random.randint(0, max_y)

                overlay_resized = overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)
                background.paste(overlay_resized, (pos_x, pos_y), overlay_resized)

                # --- 3. Calculer BBox et Label YOLO ---
                bbox = (pos_x, pos_y, pos_x + new_width, pos_y + new_height)
                try:
                    yolo_coords = _convert_to_yolo_bbox(bg_width, bg_height, bbox)
                except ValueError as e_yolo:
                     print(f"Erreur [OverlayPair]: Erreur conversion YOLO pour {overlay_path.name} sur {background_path.name}: {e_yolo}")
                     return None

                yolo_label_str = f"{yolo_class_id} {yolo_coords[0]:.6f} {yolo_coords[1]:.6f} {yolo_coords[2]:.6f} {yolo_coords[3]:.6f}"

                # Succès ! Retourner seulement l'image et le label
                return {
                    'image': background,
                    'label': yolo_label_str,
                }

        print(f"Avertissement [OverlayPair]: Impossible de placer {overlay_path.name} sur {background_path.name} après {max_attempts} tentatives.")
        return None

    except FileNotFoundError as e:
        print(f"Erreur [OverlayPair]: Fichier non trouvé: {e}")
        return None
    except UnidentifiedImageError as e:
        print(f"Erreur [OverlayPair]: Impossible d'ouvrir l'image {e}")
        return None
    except Exception as e:
        print(f"Erreur [OverlayPair] inattendue traitant {overlay_path.name} sur {background_path.name}: {e}")
        return None