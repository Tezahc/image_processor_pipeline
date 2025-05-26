import cv2
import random
import numpy as np
from warnings import warn
from pathlib import Path
from typing import Any, List, Optional, Tuple
from image_processor_pipeline.utils import utils
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn
from icecream import ic


def _load_image(filepath: Path) -> np.ndarray:
    """Charge une image avec OpenCV.

    Parameters
    ----------
    filepath : Path
        Chemin vers l'image.

    Returns
    -------
    np.ndarray
        Image BGR.
    
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    IOError
        Si OpenCV n'arrive pas à lire l'image.
    """
    if not filepath.isfile():
        raise FileNotFoundError(f"Image non trouvée: {filepath}")
    img = cv2.imread(str(filepath))
    if img is None:
        raise IOError(f"Impossible de charger l'image {filepath.name} via OpenCV.")
    return img

def _read_bboxes(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Lit un fichier de labels YOLO (.txt) et renvoie les classes et bboxes.

    Parameters
    ----------
    filepath : Path
        Chemin du fichier `.txt`

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - classes: shape (N, 1), dtype=int
        - bboxes: shape (N, 4), format [cx, cy, w, h] normalisés
    
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le contenu est invalide.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Fichier label non trouvé : {filepath}")
    data = np.loadtxt(filepath, ndmin=2)
    try:
        classes = data[:, 0].astype(int)
        bboxes = data[:, 1:5].astype(float)
    except Exception as e:
        raise ValueError(f"Format invalide dans {filepath.name}: {e}")
    return classes, bboxes

def _save_crop_files(
    img: np.ndarray,
    labels: Tuple[np.ndarray, np.ndarray],
    img_out: Path,
    label_out: Path
) -> None:
    """Sauvegarde l'image et les labels associés.

    Parameters
    ----------
    img : np.ndarray
        Image à sauvegarder.
    labels : Tuple[np.ndarray, np.ndarray]
        Classes (N, 1) et bboxes normalisées (N, 4)
    img_out : Path
        Chemin du fichier image de sortie.
    label_out : Path
        Chemin du fichier label de sortie.
    
    Raises
    ------
    IOError
        Si l'image ne peut être écrite.
    """
    classes, bboxes = labels
    if not cv2.imwrite(str(img_out), img):
        raise IOError(f"Échec écriture de l'image : {img_out}")
    
    with open(label_out, 'w', encoding='utf-8') as f:
        for cls_id, box in zip(classes, bboxes):
            cx, cy, w, h = box
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def process_square_crop_around_bbox(
    input_image_path: Path,
    input_label_path: Path,
    output_dirs: List[Path],
    # options utiles
    **options: Any # Accepter d'autres options non utilisées
) -> Optional[List[Path]]: # Retourne une liste de 2 Path (image, label) ou None
    """
    Croppe une image en carré aléatoire autour de ses bboxes et sauvegarde le résultat.

    L'image finale est un carré de côté `min(largeur, hauteur)` de l'image d'origine,
    positionné aléatoirement pour contenir toutes les annotations.

    Parameters
    ----------
    input_image_path : Path
        Chemin du fichier image d'entrée.
    input_label_path : Path
        Chemin du fichier de labels YOLO.
    output_dirs : List[Path]
        [répertoire images, répertoire labels].
    **options : Any
        Accepte d'autres options (non utilisées ici).

    Returns
    -------
    List[Path]
        Liste contenant [chemin_image_crop, chemin_label_crop]

    Raises
    ------
    IndexError
        Si moins de deux répertoires de sortie.
    FileNotFoundError
        Si image ou label d'entrée manquant.
    IOError
        Si l'image ne peut être lue ou écrite.
    ValueError
        Si format de label invalide.
    RuntimeError
        Si aucune position de crop valide trouvée.

    Examples
    --------
    >>> process_square_crop_around_bbox(
    ...     Path('img.jpg'), Path('img.txt'), [Path('out/imgs'), Path('out/labels')]
    )
    [Path('out/imgs/crop_img.jpg'), Path('out/labels/crop_img.txt')]
    """
    # --- 1. Validation des chemins ---
    image_target_dir, label_target_dir = utils._validate_dirs(output_dirs)

    if input_image_path.stem != input_label_path.stem:
        warn(f"Warning [Crop Carré]: image ({input_image_path.name}) et label ({input_label_path.name}) "
             "n'ont pas le même nom. Fichier ignoré et poursuite du traitement...")
    
    # --- 2. Chargement Image et Label ---
    image = _load_image(input_image_path)
    class_ids, bboxes = _read_bboxes(input_label_path)
    height, width = image.shape[:2]

    # --- 3. Conversion bbox noramlisées -> absolues ---
    # (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
    bboxes_absolute = xywhn2xyxy(bboxes, width, height)

    # --- 4. Logique du Crop Carré Aléatoire autour de la BBox ---
    # Taille du crop carré
    # NOTE: Changer ici pour faire un crop plus resserré
    crop_size = min(height, width)
    # récupère les dimensions extrêmes de toutes les bbox pour avoir la zone à conserver
    x_min, y_min = bboxes_absolute[:, :2].min(axis=0)
    x_max, y_max = bboxes_absolute[:, 2:].max(axis=0)

    # Calculer les bornes pour le coin supérieur gauche (x0, y0) du crop
    # pour que la bbox soit contenue.
    lower_bound_x = max(0, int(x_max - crop_size))
    upper_bound_x = min(int(x_min), width - crop_size)
    lower_bound_y = max(0, int(y_max - crop_size))
    upper_bound_y = min(int(y_min), height - crop_size)
    
    # Vérifier s'il existe une position valide
    if lower_bound_x > upper_bound_x or lower_bound_y > upper_bound_y:
        raise RuntimeError(
            f"Impossible de trouver une position de crop carré valide "
            f"contenant entièrement la bbox [{x_min},{y_min},{x_max},{y_max}] "
            f"dans une image {width}x{height} avec crop_size={crop_size}. Crop annulé.")
    
    # Choisir une position aléatoire valide
    x0 = random.randint(lower_bound_x, upper_bound_x)
    y0 = random.randint(lower_bound_y, upper_bound_y)

    # --- 5. Crop ---
    cropped_image = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    if cropped_image.size == 0:
        raise RuntimeError(f"Le crop a produit une image vide.")
    
    # --- 6. Recalibrage des bboxes sur l'image crop ---
    shifted = bboxes_absolute - np.array([[x0, y0, x0, y0]])
    clipped = np.zeros_like(shifted)

    # clip permet de caler les coordonnées entre 2 bornes
    clipped[:, 0] = np.clip(shifted[:, 0], 0, crop_size)  # all x1
    clipped[:, 1] = np.clip(shifted[:, 1], 0, crop_size)  # all y1
    clipped[:, 2] = np.clip(shifted[:, 2], 0, crop_size)  # all x2
    clipped[:, 3] = np.clip(shifted[:, 3], 0, crop_size)  # all y2

    valid = (clipped[:, 0] < clipped[:, 2]) and (clipped[:, 1] < clipped[:, 3])
    if not any(valid):
        raise RuntimeError(f"Aucune bbox résiduelle après le crop.")
    
    # ne garde que les bbox "valides" et les convertis au format yolo normalisé
    new_bboxes_absolute = clipped[valid]
    new_class_ids = class_ids[valid]
    new_bboxes = xyxy2xywhn(new_bboxes_absolute, crop_size, crop_size)

    # --- 7. Sauvegarde Image et Label ---
    img_output_path = image_target_dir / input_image_path.name
    label_output_path = label_target_dir / input_label_path.name
    _save_crop_files(cropped_image, (new_class_ids, new_bboxes), img_output_path, label_output_path)

    return [img_output_path, label_output_path]

if __name__ == '__main__':
    process_square_crop_around_bbox(
        Path('crop_carre_test/imgs/AUTOFLUSH32.jpg'),
        Path('crop_carre_test/labels/CARESITE38.txt'),
        ["CropCarre/imgs", "CropCarre/labels"]
    )
