import cv2
import logging
import math
from pathlib import Path
import random
from typing import Any, List, Optional, Tuple
from warnings import warn

import albumentations as A
from icecream import ic
import numpy as np
from PIL import Image, ImageOps
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn

from ipp.utils import utils
from ipp.utils.artifact import Artifact
from ipp.utils.yolo_labels import YoloLabelHandler, YoloLabel, BBoxLabel, SegmentationLabel


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


def crop_around_poi(
    image_path: Path, 
    seg_label_path: Path, 
    bbox: list, 
    filename_suffix: str,
    output_dirs: Path, 
    ratio: float = 1.0, 
    margin: int = 0,
    fixed_base_size: int = None
) -> List[Artifact]:
    """
    Applique un rognage carré centré sur un point d'intérêt simultanément à une image et son masque.
    
    :param bbox: [xmin, ymin, xmax, ymax]
    """
    # 1. Lecture image et masque (gestion des chemins sécurisée avec Pathlib)
    # L'image est chargée en RGB car Albumentations travaille de base de manière optimale en RGB
    # image = cv2.imread(str(image_path))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    out_img_dir, out_mask_dir = utils._validate_dirs(output_dirs, 2)

    image = Image.open(image_path)
    ImageOps.exif_transpose(image, in_place=True)
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    image_np = np.array(image)
    w_img, h_img = image.size
    
    # Le masque est chargé en niveau de gris (1 seul canal)
    lbl_handler = YoloLabelHandler(w_img, h_img)
    lbl_handler.load(seg_label_path)

    xc_norm, yc_norm, w_norm, h_norm = bbox
    cx = xc_norm * w_img
    cy = yc_norm * h_img
    w_box = w_norm * w_img
    h_box = h_norm * h_img

    # 3. Calcul de la taille de base (largeur/hauteur de la bbox ou taille fixe)
    if fixed_base_size:
        base_size = fixed_base_size
    else:
        base_size = max(w_box, h_box)
        
    # 4. Paramétrage "save_one_box" : Taille finale du côté du carré
    final_size = int(base_size * ratio + 2 * margin)
    half_size = final_size / 2.0
    
    # 5. Coordonnées théoriques du crop
    crop_xmin = int(cx - half_size)
    crop_xmax = int(cx + half_size)
    crop_ymin = int(cy - half_size)
    crop_ymax = int(cy + half_size)
    
    # 6. Gestion des débordements (Padding) si le centre est proche des bords
    # On calcule combien de pixels manquent pour faire un vrai carré complet
    pad_top = max(0, -crop_ymin)
    pad_bottom = max(0, crop_ymax - h_img)
    pad_left = max(0, -crop_xmin)
    pad_right = max(0, crop_xmax - w_img)
    
    # 7. Pipeline Albumentations (Version 2.0.8)
    transform = A.Compose([
        # Étape A: On pad l'image si les limites sortent de l'image source (avec du noir / 0)
        A.PadIfNeeded(
            min_height=h_img + pad_top + pad_bottom,
            min_width=w_img + pad_left + pad_right,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,      # Valeur de remplissage image
            fill_mask=0  # Valeur de remplissage masque
        ),
        # Étape B: Rognage exact du carré maintenant que les bordures sont sécurisées
        A.Crop(
            x_min=crop_xmin + pad_left,
            y_min=crop_ymin + pad_top,
            x_max=crop_xmax + pad_left,
            y_max=crop_ymax + pad_top
        )
    ])

    masks, seg_classes = lbl_handler.as_masks()
    if masks:
        augmented = transform(image=image_np, masks=masks)
        new_masks = augmented["masks"]
    else:
        # l'image n'a pas de label associé
        augmented = transform(image=image_np)
        new_masks = []
    
    cropped_image = augmented['image']
    lbl_handler.update_from_masks(new_masks, seg_classes)

    # 8. Préparation des chemins de sauvegarde
    out_img_path = out_img_dir / image_path.with_stem(f"{image_path.stem}_{filename_suffix}").name
    out_label_path = out_mask_dir / seg_label_path.with_stem(f"{seg_label_path.stem}_{filename_suffix}").name
    lbl_handler.save(out_label_path)
    
    # 9. Sauvegarde (retour en BGR pour OpenCV)
    Image.fromarray(cropped_image).save(out_img_path, quality=95)
    # cv2.imwrite(str(out_img_path), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    
    artifact = Artifact(
        image_path=out_img_path,
        transformation=crop_around_poi.__name__,
        params={
            "original_bbox": bbox,
            "center": [cx, cy],
            "crop_ratio": ratio,
            "crop_margin": margin,
            "final_size": final_size,
            "padding_applied": {
                "top": pad_top, "bottom": pad_bottom, 
                "left": pad_left, "right": pad_right
            }
        },
        label_path=out_label_path,
        extra={
            "source_image": image_path.name,
            "source_label": seg_label_path.name,
            "polygons_found":len(lbl_handler)
        }
    )

    return [artifact]