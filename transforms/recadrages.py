from pathlib import Path
from typing import Any, List, Optional, Tuple
import cv2


def _compute_crop(value, total_length):
    if value < 0:
        raise ValueError("Les valeurs de rognage ne peuvent pas être négatives.")
    return int(total_length * value) if 0 <= value < 1 else int(value)


def crop_from_border(
        file: Path, 
        output_dirs: List[Path], 
        crop_margins: Tuple[float, float, float, float] = (0, 0, 0, 0),
        **options: Any
    ) -> Optional[Path]:

    output_dir = output_dirs[0]

    # Filtrer les fichiers .jpg
    if file.suffix.lower() not in ('.jpg', '.jpeg'):
        raise ValueError(f"Le Fichier {file.name} n'est pas du type JPG.")

    crop_top, crop_bottom, crop_left, crop_right = crop_margins

    # lecture de l'image
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image {file.name}.")

    # Dimensions de l'image
    height, width = image.shape[:2]

    # Calcul des pixels à recadrer
    crop_top_px = _compute_crop(crop_top, height)
    crop_bottom_px = _compute_crop(crop_bottom, height)
    crop_left_px = _compute_crop(crop_left, width)
    crop_right_px = _compute_crop(crop_right, width)

    if crop_top_px + crop_bottom_px >= height or crop_left_px + crop_right_px >= width:
        raise ValueError(f"Les marges de rognage sont trop grandes pour l'image {file.name}.")

    # Recadrage
    cropped_image = image[crop_top_px:height - crop_bottom_px, crop_left_px:width - crop_right_px]

    output_path = output_dir / file.name

    try:
        success = cv2.imwrite(str(output_path), cropped_image)
        if success:
            return output_path
        else:
            # L'écriture a échoué sans lever d'exception (rare mais possible)
            print(f"Avertissement [{file.name} - Symétrie]: Échec de sauvegarde (imwrite a retourné False) pour {output_path.name}")
            return None
    except Exception as e_save:
        # Erreur lors de l'écriture (permissions, disque plein, etc.)
        print(f"Erreur [{file.name} - Symétrie]: Échec de sauvegarde pour {output_path.name}: {e_save}")
        return None

