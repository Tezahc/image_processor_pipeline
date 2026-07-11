"""
conversion_yolo.py — Conversion de masques RLE (Label Studio) en labels YOLO segmentation.
 
Conçu pour s'intégrer dans un DataFrameProcessingStep.
Le DataFrame attendu (pré-agrégé, une ligne = une image) :
 
    image_path      | rle_data                      | brushlabels
    Path ou str     | List[List[int]]  (N masques)  | List[str]  (N labels, même ordre que rle_data)
 
Exemple de préparation du DataFrame (agrégation depuis l'export Label Studio,
une ligne par région -> une ligne par image) :
 
    df_agg = (
        df.groupby("image")
        .agg(rle_data=("rle", list), brushlabels=("brushlabels", list))
        .reset_index()
        .rename(columns={"image": "image_name"})
    )
    df_agg["image_path"] = image_pool / df_agg["image_name"]
 
Exemple d'utilisation dans un pipeline :
 
    step = DataFrameProcessingStep(
        name="rle_to_yolo",
        source=df_agg,
        path_cols=["image_path"],
        data_cols=["rle_data", "brushlabels"],
        output_dirs=[images_out_dir, labels_out_dir],
        process_function=convert_rle_regions_to_yolo,
        options={"label_map": {"catheter": 1, "wire": 2}, "tol": 0.002},
        save_log=True,   # manifest JSON — inclut les params de reconstruction de chaque région
    )
"""


import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from label_studio_converter.brush import decode_rle
import numpy as np
import pandas as pd

from deepcath import utils as dutils
from ipp.utils import utils
from ipp.utils.artifact import Artifact


logger = logging.getLogger(__name__)


class YoloSegmentationConverter:
    def __init__(self, image_name :str|Path):
        self.image_name = image_name
        self.img, self.w, self.h = dutils.load_image(self.image_name)
        self.yolo_lines = []

    def apply_crop(self, x_offset :int, y_offset :int, final_width :int, final_height :int):
        """
        Définit une nouvelle base de travail en croppant l'image actuelle.
        Cette méthode écrase self.img, w et h
        """
        # Crop PIL : (left, top, right, bottom)
        self.img = self.img.crop((x_offset, y_offset, x_offset+final_width, y_offset+final_height))
        self.w, self.h = self.img.size
        logger.info(f"Crop appliqué : {final_width}x{final_height} à partir de ({x_offset}, {y_offset})")

        # checkpoint visuel du nouveau canvas
        logger.debug(f"Nouveau canvas après crop", extra={"image": np.array(self.img)})

    def rle_to_mask(self, rle_raw_data :List[int]) -> np.ndarray:
        """
        Etape 0: Décodage RLE vers masque binaire
        Entrée : Liste RLE d'entier [0-255]
        Sortie : Masque binaire de présence ou non du masque
        """
        mask_decoded = decode_rle(rle_raw_data)
        try: 
            mask_rgba = np.reshape(mask_decoded, (self.h, self.w, 4))
        except ValueError as ve:
            logger.error(f"Incompatibilité de dismensions : {ve}")
            raise ve

        # On ne garde que le canal alpha pour obtenir le masque
        mask_alpha = mask_rgba[..., 3]
    
        # On binarise le masque : toute valeur > 0 devient True
        binary_mask = (mask_alpha > 0)

        logger.debug(f"Etape 0: masque binaire (RLE Decoded)", extra={"image": binary_mask})
        return binary_mask

    def mask_to_contours(self, mask: np.ndarray):
        """
        Etape 1: Extraction des contours du masque binaire via OpenCV.
        Entrée: mask (numpy array uint8, 0 ou 255)
        Sortie: list of numpy arrays au format OpenCV (N, 1, 2)
         
           list de listes de points [ [x1, y1, x2, y2, ...], [...] ]
        """
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask.astype(np.uint8)

        # RETR_EXTERNAL: on ne veut que le contour extérieur
        # CHAIN_APPROX_SIMPLE: compresse les segments horizontaux/verticaux
        # CHAIN_APPROX_NONE: à préférer si simplification avec approxPolyDP ensuite ?
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if logger.isEnabledFor(logging.DEBUG):
            debug_img = np.array(self.img.convert("RGB"))
            cv2.drawContours(debug_img, contours, -1, (255, 0, 0), 5)
            logger.debug("Contours extraits", extra={"image": debug_img})

        return contours
    
    def contours_to_polygons(self, contours: List[np.ndarray], min_area=10):
        """
        Etape 2: Conversion du format OpenCV vers une liste de polygones standards.
        Filtre également les contours trop petits (bruit).
        Entrée: liste d'arrays (N, 1, 2)
        Sortie liste d'arrays (N, 2)
        """
        polygons = []
        for cnt in contours:
            # Filtrage par taille (évite les résidus de brosses ou le bruit)
            if cv2.contourArea(cnt) < min_area:
                continue

            # Aplatit le format de (N, 1, 2) vers (N, 2)
            polygons.append(cnt.reshape(-1, 2))

        # "opti"
        # return [c.reshape(-1, 2) for c in contours if cv2.contourArea(c) > min_area]
        return polygons
    
    def simplify_polygons(self, polygons: List[np.ndarray], tolerance=0.001):
        """
        Etape 3: Simplification des polygones par optimisation du nombre de points.
        Entrée: list of arrays (N, 2)
        Sortie: list of arrays (M, 2) avec M <= 2
        """
        simplified :List[np.ndarray]= []
        for poly in polygons:
            # arcLength calcule le périmètre. 
            # epsilon est la distance max de déviation autorisée.
            epsilon = tolerance * cv2.arcLength(poly, True)
            # approxPolyDP réduit le nombre de sommets
            approx :np.ndarray = cv2.approxPolyDP(poly, epsilon, True)
            # cnt est de forme (N, 1, 2), on l'aplatit en (N, 2)
            simplified.append(approx.reshape(-1, 2))
            
        if logger.isEnabledFor(logging.DEBUG):
            viz_simpl = np.zeros((self.h, self.w), dtype=np.uint8)
            cv_format = [p.reshape(-1, 1, 2) for p in simplified]
            cv2.polylines(viz_simpl, cv_format, isClosed=True, color=255, thickness=3)
            logger.debug(f"Polygones après simplification (tol={tolerance})", extra={"image": viz_simpl})
            
        return simplified

    def normalize_polygons(self, polygons :List[np.ndarray]):
        """
        Etape 4: Normalisation (0-1) et formatage YOLO.
        Entrée: Liste de numpy arrays (N, 2) en pixels
        Sortie: Liste de numpy arrays (N, 2) en float relatives
        """
        normalized = []
        for poly in polygons:
            # Normalisation
            poly_norm = poly.astype(float)
            poly_norm[:, 0] /= self.w  # x / W
            poly_norm[:, 1] /= self.h  # y / H
            normalized.append(poly_norm)
            
        return normalized
    
    def _visualize_yolo(self, yolo_lines :List[str], class_names :dict = None):
        """
        Check de debug final: Redessine les polygones yolo sur l'image originale en utilisant l'Annotator d'ultralytics.
        """
        img_bgr = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)

        for line in yolo_lines:
            parts = line.split()
            # cls_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= self.w
            coords[:, 1] *= self.h
            coords = coords.astype(int)

            pts = coords.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [pts], isClosed=True, color=(0,255,0), thickness=3)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        return img
    
    def format_to_yolo(self, normalized_polygons :List[np.ndarray], class_id :int) -> List[str]:
        """
        Etape 5: Génération des lignes de texte format YOLO.
        Entrée: list d'arrays (N, 2) relatives
        Sortie: liste de strings "id x1 x2 ... xn yn"
        """
        lines = []
        for poly in normalized_polygons:
            # Aplatissement: [[x1,y1], [x2,y2]] -> [x1, y1, x2, y2]
            flat_coords = poly.flatten().tolist()
            # Création de la ligne: class_id x1 y1 x2 y2 ...
            coords_str = " ".join([f"{c:.6f}" for c in flat_coords])
            lines.append(f"{class_id} {coords_str}")
        
        logger.debug(f"Yolo lines sur l'image", extra={"image": self._visualize_yolo(lines)})
        return lines
    
    def save_img(self, output_path:Path):
        self.img.save(output_path)
    
    def write_yolo_lines(self, output_path:Path):
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(self.yolo_lines))
        logger.info(f"Fichier créé : {output_path}")
        
    def convert(self, rle_data, class_id, tol=0.001, simplify=True):
        """Pipeline complet pour un masque donné"""
        # 0. RLE -> Mask
        mask = self.rle_to_mask(rle_data)

        # 1. Contours
        raw_contours = self.mask_to_contours(mask)
        
        # 2. Polygons
        polygons = self.contours_to_polygons(raw_contours)

        # 3. Simplification
        if simplify:
            polygons = self.simplify_polygons(polygons, tol)
        
        # 4. Normalisation
        normalized_polys = self.normalize_polygons(polygons)

        # 5. Formatage
        new_lines = self.format_to_yolo(normalized_polys, class_id)
        self.yolo_lines.extend(new_lines)
        
        return self.yolo_lines


def convert_rle_regions_to_yolo(
    image_path: Path,
    rle_data: List[List[int]],
    brushlabels: List[str],
    output_dirs: List[Path],
    *,
    label_map: Dict[str, int],
    tol: float = 0.001,
    simplify: bool = True,
    **kwargs,
) -> Optional[List[Artifact]]:
    """Convertit toutes les régions RLE d'une image en un unique fichier de labels YOLO segmentation.

    Conçue pour être appelée par ``DataFrameProcessingStep`` — chaque appel traite
    une image et l'ensemble de ses régions annotées (jointure 1→N déjà résolue en
    amont, lors de la préparation du DataFrame : une ligne = une image,
    ``rle_data`` et ``brushlabels`` sont des listes de même longueur).

    Parameters
    ----------
    image_path : Path
        Image source, utilisée pour dimensionner les masques décodés
        (``YoloSegmentationConverter`` s'appuie sur ``dutils.load_image``).
    rle_data : List[List[int]]
        Liste des données RLE brutes (une par région annotée sur l'image).
    brushlabels : List[str]
        Libellés de classe associés, **même ordre et même longueur** que ``rle_data``.
    output_dirs : List[Path]
        ``[0]`` → dossier de sortie des images YOLO (créé par le pipeline).
        ``[1]`` → dossier de sortie des labels YOLO (créé par le pipeline).
    label_map : Dict[str, int]
        Correspondance ``brushlabels -> class_id`` YOLO. Un libellé absent de ce
        dictionnaire reçoit l'id ``0``.
    tol : float, default 0.001
        Tolérance de simplification polygonale (epsilon relatif au périmètre),
        transmise à ``simplify_polygons``.
    simplify : bool, default True
        Si ``False``, l'étape de simplification des polygones est sautée.

    Returns
    -------
    List[Artifact]
        Liste à un élément. ``Artifact.image_path`` pointe vers le fichier image
        produit (``output_dirs[0]``), et ``Artifact.label_path`` pointe vers le
        fichier ``.txt`` YOLO produit (``output_dirs[1]``). 
        ``params`` contient ``tol``, ``simplify`` et, pour chaque région 
        effectivement convertie, son ``brushlabels``, ``class_id`` résolu
        et ``rle`` brut : de quoi rejouer la conversion à l'identique sans DataFrame.

    None
        Si aucune région n'est fournie, ou si toutes échouent à la conversion
        (aucune ligne YOLO générée).

    Raises
    ------
    ValueError
        Si ``rle_data`` et ``brushlabels`` n'ont pas la même longueur — signe
        d'une désynchronisation lors de l'agrégation du DataFrame en amont.

    Notes
    -----
    Une image sans annotation ne devrait jamais atteindre cette fonction dans le
    flux normal (l'agrégation ``groupby("image")`` ne produit une ligne que pour
    les images ayant au moins une région) ; si ``rle_data`` est malgré tout vide,
    aucune ligne YOLO n'est produite et la fonction retourne ``None``.

    L'échec de chargement de l'image (fichier manquant, format invalide, etc.)
    n'est pas intercepté ici : l'exception remonte telle quelle et le pipeline
    la capture nativement, l'enregistrant avec le statut ``"Error"`` et son
    message — pas besoin de dupliquer cette logique dans la fonction.

    Examples
    --------
        >>> from pathlib import Path
    >>> artifacts = convert_rle_regions_to_yolo(
    ...     image_path=Path("images/img_001.jpg"),
    ...     rle_data=[[0, 255, 0, 255]],          # un RLE brut par région
    ...     brushlabels=["catheter"],
    ...     output_dirs=[Path("dataset/images"), Path("dataset/labels")],
    ...     label_map={"catheter": 1, "wire": 2},
    ... )
    >>> artifacts[0].image_path  # doctest: +SKIP
    PosixPath('dataset/images/img_001.jpg')
    >>> artifacts[0].label_path  # doctest: +SKIP
    PosixPath('dataset/labels/img_001.txt')
    """
    image_dir, label_dir = utils._validate_dirs(output_dirs, nb_dirs=2)

    if len(rle_data) != len(brushlabels):
        raise ValueError(
            f"[{image_path.name}] rle_data ({len(rle_data)}) et brushlabels "
            f"({len(brushlabels)}) de longueurs différentes — vérifier "
            f"l'agrégation du DataFrame en amont (groupby)."
        )

    # Le chargement de l'image n'est pas protégé : une erreur ici remonte au
    # pipeline qui la logue nativement en statut "Error" (cf. Notes).
    converter = YoloSegmentationConverter(image_path)

    reconstruction_regions = []

    for rle, brushlabel in zip(rle_data, brushlabels):
        class_id = label_map.get(brushlabel, 0)

        try:
            converter.convert(rle_data=rle, class_id=class_id, tol=tol, simplify=simplify)
        except Exception as e:
            # Une région défectueuse ne bloque pas les autres régions de l'image
            logger.error(f"[{image_path.name}] Échec conversion région '{brushlabel}' : {e}")
            continue

        reconstruction_regions.append({
            "brushlabels": brushlabel,
            "class_id": class_id,
            "rle": rle,
        })

    if not converter.yolo_lines:
        logger.warning(f"[{image_path.name}] Aucune ligne YOLO générée (aucune région fournie ou toutes en échec).")
        return None

    label_path = label_dir / f"{image_path.stem}.txt"
    converter.write_yolo_lines(label_path)
    
    image_path_out = image_dir / image_path.name
    converter.save_img(image_path_out)

    artifact = Artifact(
        image_path=image_path_out,
        transformation=convert_rle_regions_to_yolo.__name__,
        params={
            "tol": tol,
            "simplify": simplify,
            "regions": reconstruction_regions,
        },
        label_path=label_path
    )

    return [artifact]