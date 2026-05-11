from pathlib import Path
from typing import Any, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from ultralytics.utils.ops import xywhn2xyxy

from ..utils.artifact import Artifact
from ..utils import utils


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
    classes, bboxes = utils._read_bboxes(label_path)
    
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


def _compute_crop(value: float, total_length: int) -> int:
    """Convertit une marge de rognage en pixels.

    Parameters
    ----------
    value : float
        Marge à appliquer.
        - 0 <= value < 1 : interprété comme un pourcentage de `total_length`.
        - value >= 1      : interprété comme un nombre de pixels fixe.
    total_length : int
        Dimension de référence (hauteur ou largeur) pour le calcul du pourcentage.

    Raises
    ------
    ValueError
        Si `value` est strictement négatif.
    """
    if value < 0:
        raise ValueError(f"Les valeurs de rognage ne peuvent pas être négatives (reçu : {value}).")
    return int(total_length * value) if value < 1 else int(value)


def zoom_crop(
    image_path: Path,
    label_path: Path,
    output_dirs: List[Path],
    crop_margins: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    min_bbox_visibility: float = 0.1,
    **options: Any,
) -> Optional[List[Artifact]]:
    """Rogne une image par ses bords avec gestion des labels YOLO associés.

    Découpe `crop_margins` sur chacun des quatre bords et recalcule les
    annotations YOLO en conséquence. Les bbox dont la surface visible après
    crop est inférieure à `min_bbox_visibility` sont supprimées.

    Ce recadrage est dit « centré » lorsque des marges symétriques sont
    fournies — par exemple ``(0.25, 0.25, 0.25, 0.25)`` pour conserver
    exactement 50 % de l'image originale — mais des marges asymétriques
    sont tout autant supportées.

    Parameters
    ----------
    image_path : Path
        Chemin du fichier image d'entrée.
    label_path : Path
        Chemin du fichier de labels YOLO associé (.txt, format cx cy w h normalisé).
    output_dirs : List[Path]
        Liste d'au moins deux répertoires : ``[images_dir, labels_dir]``.
    crop_margins : Tuple[float, float, float, float]
        Marges de rognage dans l'ordre ``(top, bottom, left, right)``.

        - ``0 <= v < 1`` → pourcentage de la dimension correspondante.
          Exemple : ``0.25`` retire 25 % de la hauteur/largeur de ce côté.
        - ``v >= 1``      → nombre de pixels fixes (converti en ``int``).

        Pour un crop centré conservant 50 % : ``(0.25, 0.25, 0.25, 0.25)``.
        Pour retirer 10 px en haut uniquement : ``(10, 0, 0, 0)``.
    min_bbox_visibility : float, default=0.1
        Fraction minimale de la surface d'une bbox qui doit rester visible
        après crop pour qu'elle soit conservée dans les labels de sortie.
        Entre 0 (tout garder) et 1 (exiger bbox intacte). Défaut : 0.1.
    **options : Any
        Arguments supplémentaires ignorés (compatibilité pipeline).

    Returns
    -------
    Optional[List[Artifact]]
        Liste contenant un unique :class:`Artifact` avec les chemins image
        et label de sortie, ainsi que les paramètres du crop.
        Retourne ``None`` si l'image ne peut pas être chargée.

    Raises
    ------
    FileNotFoundError
        Si l'image ou le label d'entrée est introuvable.
    IndexError
        Si ``output_dirs`` contient moins de deux éléments.
    ValueError
        Si les marges dépassent les dimensions de l'image, ou sont négatives.
    IOError
        Si l'écriture de l'image de sortie échoue.

    Examples
    --------
    Crop centré conservant 50 % de l'image (25 % supprimés sur chaque bord) :

    >>> artifacts = zoom_crop(
    ...     image_path=Path("data/imgs/img001.jpg"),
    ...     label_path=Path("data/labels/img001.txt"),
    ...     output_dirs=[Path("out/imgs"), Path("out/labels")],
    ...     crop_margins=(0.25, 0.25, 0.25, 0.25),
    ... )

    Retirer 80 px en haut et 40 px à gauche, rien ailleurs :

    >>> artifacts = zoom_crop(
    ...     image_path=Path("data/imgs/img001.jpg"),
    ...     label_path=Path("data/labels/img001.txt"),
    ...     output_dirs=[Path("out/imgs"), Path("out/labels")],
    ...     crop_margins=(80, 0, 40, 0),
    ... )
    """
    # --- 1. Validation des répertoires de sortie ---
    if len(output_dirs) < 2:
        raise IndexError(
            f"zoom_crop requiert au moins 2 répertoires de sortie "
            f"[images, labels], reçu : {len(output_dirs)}."
        )
    out_image_dir, out_label_dir = utils._validate_dirs(output_dirs, 2)

    # --- 2. Chargement de l'image ---
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path.name}")
    height, width = image.shape[:2]

    # --- 3. Calcul des marges en pixels ---
    crop_top, crop_bottom, crop_left, crop_right = crop_margins
    top_px    = _compute_crop(crop_top,    height)
    bottom_px = _compute_crop(crop_bottom, height)
    left_px   = _compute_crop(crop_left,   width)
    right_px  = _compute_crop(crop_right,  width)

    if top_px + bottom_px >= height:
        raise ValueError(
            f"Marges verticales ({top_px} + {bottom_px} px) ≥ hauteur "
            f"({height} px) pour {image_path.name}."
        )
    if left_px + right_px >= width:
        raise ValueError(
            f"Marges horizontales ({left_px} + {right_px} px) ≥ largeur "
            f"({width} px) pour {image_path.name}."
        )

    # --- 4. Chargement des labels YOLO ---
    if not label_path.exists():
        raise FileNotFoundError(f"Label introuvable : {label_path}")
    class_ids, bboxes = utils._read_bboxes(label_path)  # bboxes : (N, 4) float, format YOLO normalisé

    # --- 5. Crop image + recalcul des bbox via Albumentations ---
    # Albumentations gère nativement le format YOLO (cx, cy, w, h normalisé)
    # et filtre les bbox dont la surface visible < min_bbox_visibility.
    transform = A.Compose(
        [
            A.Crop(
                x_min=left_px,
                y_min=top_px,
                x_max=width  - right_px,
                y_max=height - bottom_px,
            )
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=min_bbox_visibility,
        ),
    )

    result = transform(
        image=image,
        bboxes=bboxes.tolist(),
        class_labels=class_ids.tolist(),
    )

    cropped_image  = result["image"]
    new_bboxes     = result["bboxes"]       # List[Tuple[cx, cy, w, h]]
    new_class_ids  = result["class_labels"]

    # --- 6. Sauvegarde image ---
    out_image_path = out_image_dir / image_path.name
    if not cv2.imwrite(str(out_image_path), cropped_image):
        raise IOError(f"Échec écriture image : {out_image_path}")

    # --- 7. Sauvegarde labels ---
    out_label_path = out_label_dir / label_path.name
    with out_label_path.open("w", encoding="utf-8") as f:
        for cls_id, (cx, cy, w, h) in zip(new_class_ids, new_bboxes):
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # --- 8. Construction et retour de l'Artifact ---
    artifact = Artifact(
        image_path=out_image_path,
        label_path=out_label_path,
        transformation="zoom_crop",
        params={
            "crop_px": {
                "top":    top_px,
                "bottom": bottom_px,
                "left":   left_px,
                "right":  right_px,
            },
            "original_size": {"width": width, "height": height},
            "output_size": {
                "width":  width  - left_px - right_px,
                "height": height - top_px  - bottom_px,
            }
        }
    )
    return [artifact]