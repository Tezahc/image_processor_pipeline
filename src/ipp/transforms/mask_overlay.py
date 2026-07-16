import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

from ultralytics.utils import plotting
from ultralytics.data.utils import polygons2masks

from ipp.utils.artifact import Artifact
from ipp.utils import utils


def process_mask_overlay(
    img_path: Path,
    lbl_path: Path,
    output_dirs: List[Path],
    alpha: float = 0.6,
    suffix_key: Optional[str] = None,
    **kwargs,
) -> Optional[List[Artifact]]:
    """Superpose des masques polygonaux sur une image et sauvegarde le résultat.

    Conçue pour être utilisée avec ``ProcessingStep`` en mode ``'zip'``
    (un dossier images + un dossier labels).

    La fonction de traitement attend des labels au format polygone YOLO :
    ``class_id x1 y1 x2 y2 ...`` (coordonnées normalisées).
    Les lignes vides ou malformées (< 3 valeurs) sont ignorées silencieusement.

    Parameters
    ----------
    img_path : Path
        Chemin de l'image source (BGR via OpenCV).
    lbl_path : Path
        Chemin du fichier label ``.txt`` (format polygone normalisé).
    output_dirs : List[Path]
        Dossiers de sortie. ``output_dirs[0]`` reçoit l'image annotée.
    alpha : float, optional
        Transparence des masques superposés. Défaut : ``0.6``.
    suffix_key : str or None, optional
        Suffixe ajouté au nom du fichier de sortie via ``build_output_filepath``.
        ``None`` ou ``"off"`` → conserve le nom d'origine. Défaut : ``None``.
    **kwargs
        Arguments supplémentaires ignorés (compatibilité pipeline).

    Returns
    -------
    List[Artifact] or None
        Liste contenant un ``Artifact`` décrivant l'image annotée sauvegardée.
        ``None`` si le fichier label est introuvable ou si l'écriture échoue.

    Raises
    ------
    FileNotFoundError
        Si ``img_path`` n'existe pas (levée par ``_load_image``).
    IOError
        Si l'image ne peut être lue ou écrite.
    ValueError
        Si le fichier label contient des données invalides.

    Examples
    --------
    Usage avec ``ProcessingStep`` :

    >>> step = ProcessingStep(
    ...     name="mask_overlay",
    ...     process_function=process_mask_overlay,
    ...     input_dirs=[img_dir, lbl_dir],
    ...     output_dirs=[out_dir],
    ...     pairing_method="zip",
    ...     options={"alpha": 0.5, "suffix_key": "overlay"},
    ... )
    >>> step.run()
    """
    # --- Validation des sorties ---
    output_dir: Path = utils._validate_dirs(output_dirs, nb_dirs=1)

    # --- Chargement image (BGR) ---
    img = utils._load_image(img_path)
    h_img, w_img = img.shape[:2]

    # --- Lecture des labels (format polygone normalisé) ---
    polygons: List[np.ndarray] = []
    colors: List = []

    if not lbl_path.is_file():
        # Pas de labels → pas de masques, mais on sauvegarde quand même l'image brute
        # pour ne pas casser le zip et garder une trace dans les logs
        print(f"  Avertissement : label introuvable pour {img_path.name}, image sauvegardée sans masque.")
    else:
        with lbl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = np.fromstring(line, sep=" ")
                # Une ligne valide doit avoir au moins : class_id + 1 point (x, y)
                if len(data) < 3:
                    continue
                cls_id = data[0]
                pts = data[1:].reshape(-1, 2) * [w_img, h_img]
                polygons.append(pts)
                colors.append(plotting.colors(cls_id))

    # --- Annotation ---
    annotator = plotting.Annotator(img.copy())

    if polygons:
        masks = polygons2masks(
            (h_img, w_img),
            polygons,
            color=1,
            downsample_ratio=1,
        )
        annotator.masks(masks, colors=colors, alpha=alpha)

    result: np.ndarray = annotator.result()

    # --- Sauvegarde ---
    out_path = utils.build_output_filepath(img_path, output_dir, suffix_key=suffix_key)

    if not cv2.imwrite(str(out_path), result):
        raise IOError(f"Échec écriture image : {out_path}")

    return [Artifact(
        image_path=out_path,
        transformation="mask_overlay",
        params={"alpha": alpha},
        # label_path=lbl_path if lbl_path.is_file() else None,
        extra={"n_masks": len(polygons)},
    )]
