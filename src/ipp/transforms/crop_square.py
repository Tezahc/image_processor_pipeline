import math
import cv2
import random
import numpy as np
from warnings import warn
from pathlib import Path
from typing import Any, List, Optional, Tuple
from ipp.utils import utils
from ipp.utils.artifact import Artifact
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn
from icecream import ic
import logging


logger = logging.getLogger("crop")

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
    image_target_dir, label_target_dir = utils._validate_dirs(output_dirs, 2)

    if input_image_path.stem != input_label_path.stem:
        warn(f"Warning [Crop Carré]: image ({input_image_path.name}) et label ({input_label_path.name}) "
             "n'ont pas le même nom. Fichier ignoré et poursuite du traitement...")
    
    # --- 2. Chargement Image et Label ---
    image = utils._load_image(input_image_path)
    class_ids, bboxes = utils._read_bboxes(input_label_path)
    height, width = image.shape[:2]

    # --- 3. Conversion bbox normalisées -> absolues ---
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

    valid = np.logical_and(
        (clipped[:, 0] < clipped[:, 2]),
        (clipped[:, 1] < clipped[:, 3])
    )
    if not any(valid):
        raise RuntimeError(f"Aucune bbox résiduelle après le crop.")
    
    # ne garde que les bbox "valides" et les convertis au format yolo normalisé
    new_bboxes_absolute = clipped[valid]
    new_class_ids = class_ids[valid]
    new_bboxes = xyxy2xywhn(new_bboxes_absolute, crop_size, crop_size)

    # --- 7. Sauvegarde Image et Label ---
    img_output_path = image_target_dir / input_image_path.name
    label_output_path = label_target_dir / input_label_path.name
    artifacts = Artifact(
        image_path=img_output_path,
        label_path=label_output_path,
        transformation="crop_square",
        params={
            "x0": x0,
            "y0": y0,
            "crop_size": crop_size
        }
    )
    _save_crop_files(cropped_image, (new_class_ids, new_bboxes), img_output_path, label_output_path)

    return artifacts

def bbox_diagonal_crop(
    image_path: Path,
    label_path: Path,
    output_dirs: List[Path],
    diag_range: Tuple[int, int] = (0.15, 0.30),
    seed: Optional[int] = None,
    **options: Any
) -> Optional[Artifact]:
    out_image_dir, out_label_dir = utils._validate_dirs(output_dirs, 2)

    # Gestion de la seed pour la reproductibilité
    if seed is None:
        seed = random.randint(0, 2**32-1)
    random.seed(seed)
    np.random.seed(seed)

    # Chargement image et label
    img = utils._load_image(image_path)
    img_height, img_width = img.shape[:2]
    logger.debug(f"taille d'image : {img.shape}")

    classes, bboxs = utils._read_bboxes(label_path)
    diag_img = math.hypot(img_width, img_height)
    logger.debug(f"bbox raw : {bboxs}")

    # calcul de la boite englobante
    bboxs_abs = xywhn2xyxy(bboxs, img_width, img_height)
    logger.debug(f"bbox absolues : {bboxs_abs}")

    x_min, y_min = bboxs_abs[:, :2].min(axis=0)
    x_max, y_max = bboxs_abs[:, 2:].max(axis=0)
    logger.debug(f"Global bbox: xmin={x_min} ymin={y_min} xmax={x_max} ymax={y_max}")

    global_width = x_max - x_min
    global_height = y_max - y_min
    logger.debug(f"Global dims: width={global_width} height={global_height}")

    min_crop_size = max(global_width, global_height) # hard constraint
    logger.debug(f"taille mini finale : {min_crop_size}")

    # diagonales
    diag_bbox = math.hypot(global_width, global_height)
    bbox_diags = [math.hypot(x2-x1, y2-y1) for x1, y1, x2, y2 in bboxs_abs]
    logger.debug(f"diag list {bbox_diags}")

    # contraintes sur la diagonale du crop
    d_min, d_max = diag_range
    diag_crop_min = max(d / d_max for d in bbox_diags)
    diag_crop_max = max(d / d_min for d in bbox_diags)
    logger.debug(f"range des diagonales : {diag_crop_min}-{diag_crop_max}")

    # conversion diag -> coté carré
    # soft constraints
    side_min = diag_crop_min / math.sqrt(2)
    side_max = diag_crop_max / math.sqrt(2)
    logger.debug(f"Taille des cotés min/max : {side_min}/{side_max}")

    # taille max possible dans l'image
    max_possible = min(img_width, img_height)

    # bornes idéales du crop
    crop_min = max(min_crop_size, side_min)
    crop_max = min(side_max, max_possible)
    logger.debug("Crop bounds: min=%.1f max=%.1f (min_crop=%.1f side_min=%.1f side_max=%.1f max_possible=%d)",
                 crop_min, crop_max, min_crop_size, side_min, side_max, max_possible)

    if crop_min <= crop_max:
        crop_size = random.randint(
            int(math.ceil(crop_min)),
            int(math.floor(crop_max))
        )
        logger.debug(f"Random size selected : {crop_size}")
    else:
        # fallbach sûr: on respecte la contrainte dure
        crop_size = int(math.ceil(min_crop_size))
        logger.debug(f"No valid random range, fallback crop_size={crop_size}")

    # placement du crop (random)
    x0_min = max(0, int(x_max - crop_size))
    x0_max = min(int(x_min), img_width - crop_size)

    y0_min = max(0, int(y_max - crop_size))
    y0_max = min(int(y_min), img_height - crop_size)
    logger.debug(f"coordonnées de crop min/max : {x0_min}-{x0_max} ; {y0_min}-{y0_max}")

    x0 = random.randint(x0_min, x0_max) if x0_min <= x0_max else x0_min
    y0 = random.randint(y0_min, y0_max) if y0_min <= y0_max else y0_min
    logger.debug(f"coords roll : x0={x0} y0={y0}")

    crop = img[y0 : y0+crop_size, x0 : x0+crop_size]
    logger.debug(f"taille du crop : {crop.shape}")

    logger.debug(f"{bboxs_abs}")

    # recadrage des bbox
    offset = np.array([x0, y0, x0, y0])
    bboxs_abs -= offset
    logger.debug(f"bbox_abs update : {bboxs_abs}")
    new_bboxs_norm = xyxy2xywhn(bboxs_abs, crop.shape[0], crop.shape[1])

    # Construction des paths de sortie
    out_image_path = utils.build_output_filepath(image_path, out_image_dir, **options)
    out_label_path = utils.build_output_filepath(label_path, out_label_dir, **options)

    # enregistrement et retour
    _save_crop_files(crop, (classes, new_bboxs_norm), out_image_path, out_label_path)
    artifacts = Artifact(
        image_path=out_image_path, 
        label_path=out_label_path,
        transformation="crop_random_square",
        params={"x0": x0,
                "y0": y0,
                "crop_size": crop_size, 
                "seed":seed}
    )
    return [artifacts]

if __name__ == '__main__':
    process_square_crop_around_bbox(
        Path('crop_carre_test/imgs/AUTOFLUSH32.jpg'),
        Path('crop_carre_test/labels/CARESITE38.txt'),
        ["CropCarre/imgs", "CropCarre/labels"]
    )
