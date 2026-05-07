import math
from pathlib import Path
from typing import Any, List, Optional, Tuple
import cv2
from PIL import Image
from ipp.transforms.crop_square import _read_bboxes
from ultralytics.utils.ops import xywhn2xyxy
import numpy as np


def _compute_crop(value, total_length):
    """vérifie que la marche de rognage a une valeur positive et renvoie une marge en pixels.
    Accepte un float pour rogner un % de l'image"""
    #TODO: vérifier la présence de total_length uniquement si value est un float entre 0 et 1. Sinon None par défaut
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

def fit_crop(
    image_path: Path,
    output_dirs: List[Path],
    **options: Any
) -> Optional[List[Path]]:
    """Crop une image de sorte à supprimer les pixels transparents superflus"""
    output_dir = output_dirs[0]

    image = Image.open(image_path)

    bbox = image.getbbox()
    if not bbox:
        new_image = image.copy()
    else:
        new_image = image.crop(bbox)
    
    output_path = output_dir / image_path.name
    new_image.save(output_path)

    return output_path

def crop_bbox(
    image_path: Path,
    output_dirs: List[Path],
    size: float = 0.5,
    **options: Any,
) -> Optional[List[Path]]:
    """rogne une image en plusieurs, autour des détections yolo de celle-ci.

    Parameters
    ----------
    image_path : Path
        chemin de l'image d'entrée.
    output_dirs : List[Path]
        chemin d'enregistrement des images.
    size : float
        rapport apparent entre la taille de la box de détection et la taille de l'image rognée.

    Returns
    -------
    Optional[List[Path]]
        chemins des images crées
    """
    # ouverture de l'image
    output_dirs = output_dirs[0]
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]

    # lecture des détections
    label_path = image_path.parent / "labels" / image_path.with_suffix(".txt").name
    if not label_path.exists():
        print(f"aucun fichier de label trouvé pour l'image {image_path.name}.")
        return
    classes, bboxes = _read_bboxes(label_path)
    
    # agrandir la zone de crop pour que la bbox soit `size`% de la zone
    # on agrandit les width et height des bbox (xywh)
    bboxes[:, 2:4] = bboxes[:, 2:4] / size
    # convertit en valeurs absolues mêmes si elles sont incohérentes (<0 ou >shape)
    bboxes_abs = xywhn2xyxy(bboxes, width, height)
    # "clip" aux dimensions de l'image
    bboxes_abs[:, [0,2]] = np.clip(bboxes_abs[:, [0,2]], 0, width)
    bboxes_abs[:, [1,3]] = np.clip(bboxes_abs[:, [1,3]], 0, height)

    # enregistre une image par détection
    for i, (cls, bbox) in enumerate(zip(classes, bboxes_abs)):
        crop_left, crop_top, crop_right, crop_bottom = map(int, bbox)
        detection = img[crop_top:crop_bottom, crop_left:crop_right]

        save_path = output_dirs / image_path.with_stem(f"{cls:02}-{image_path.stem}-id{i}").name
        cv2.imwrite(str(save_path), detection)
