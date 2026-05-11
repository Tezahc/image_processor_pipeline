import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from label_studio_converter.brush import decode_rle
import numpy as np
import pandas as pd

from deepcath import utils


logger = logging.getLogger(__name__)


class YoloSegmentationConverter:
    def __init__(self, image_name :str|Path):
        self.image_name = image_name
        self.img, self.w, self.h = utils.load_image(self.image_name)
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


def convert_rle_to_yolo_labels(
    image_path: Path,
    output_dirs: List[Path],
    *,
    df: pd.DataFrame,
    label_map: Dict[str, int],
    filename_col: str = "image",
    brushlabels_col: str = "brushlabels",
    rle_col: str = "rle",
    tol: float = 0.001,
    simplify: bool = True,
    save_params: bool = True,
    **kwargs,
) -> Optional[Path]:
    """Convertit les annotations RLE d'une image en fichier de labels YOLO segmentation.

    Cette fonction est conçue pour être passée comme ``process_function`` à une
    ``DataFrameImportStep``. Elle reçoit un chemin image à la fois (mode ``one_input``),
    filtre le DataFrame sur ce fichier, puis exécute la chaîne de conversion
    ``YoloSegmentationConverter`` pour chaque région annotée.

    Si ``save_params=True``, un fichier sidecar ``<stem>.params.json`` est écrit dans
    le même dossier de sortie. Il contient les hyperparamètres (``tol``, ``simplify``)
    et, pour chaque région, le ``brushlabels``, le ``class_id`` résolu et le ``rle``
    brut. Ces données sont suffisantes pour rejouer la conversion à l'identique sans
    accès au DataFrame d'origine.

    Parameters
    ----------
    image_path : Path
        Chemin absolu vers l'image à traiter, fourni par l'orchestrateur.
        Seul le ``.name`` est utilisé pour filtrer le DataFrame.
    output_dirs : List[Path]
        Liste des dossiers de sortie fournie par le pipeline.
        ``output_dirs[0]`` reçoit le ``.txt`` YOLO (et le ``.params.json`` si activé).
    df : pd.DataFrame
        DataFrame contenant *toutes* les annotations. Doit posséder au minimum les
        colonnes ``filename_col``, ``brushlabels_col`` et ``rle_col``.
        Passé via ``options={"df": df, ...}`` lors de la création de l'étape.
    label_map : Dict[str, int]
        Correspondance ``brushlabels -> class_id`` YOLO.
        Les libellés absents de ce dictionnaire reçoivent l'id ``0`` (comportement
        identique à ``labels.get(row.brushlabels, 0)`` du notebook original).
    filename_col : str, default ``"image"``
        Nom de la colonne contenant les noms de fichiers dans ``df``.
    brushlabels_col : str, default ``"brushlabels"``
        Nom de la colonne contenant le label de classe de chaque région.
    rle_col : str, default ``"rle"``
        Nom de la colonne contenant les données RLE brutes.
    tol : float, default ``0.001``
        Tolérance de simplification polygonale passée à ``simplify_polygons``
        (paramètre ``epsilon`` relatif au périmètre).
    simplify : bool, default ``True``
        Si ``False``, l'étape de simplification des polygones est sautée.
    save_params : bool, default ``True``
        Si ``True``, écrit le fichier sidecar ``.params.json`` de reconstruction.
    **kwargs
        Arguments supplémentaires ignorés silencieusement (tolérance aux options
        génériques du pipeline).

    Returns
    -------
    Path
        Chemin vers le fichier ``.txt`` YOLO créé (``output_dirs[0] / <stem>.txt``).
    None
        Si aucune annotation n'est trouvée pour cette image dans ``df``,
        ou si une erreur survient pendant la conversion.

    Notes
    -----
    **Reconstruction à l'identique**

    Le fichier ``.params.json`` sidecar contient tout le nécessaire pour rejouer
    la conversion sans le DataFrame original :

    .. code-block:: json

        {
            "image_name": "img_001.jpg",
            "tol": 0.001,
            "simplify": true,
            "regions": [
                {
                    "brushlabels": "catheter",
                    "class_id": 1,
                    "rle": [0, 255, 0, ...]
                }
            ]
        }

    **Utilisation avec DataFrameImportStep**

    .. code-block:: python

        step = DataFrameImportStep(
            name="rle_to_yolo",
            source=df_full,                     # DataFrame ou chemin .pkl / .csv
            image_pool=Path("images/"),
            filename_col="image",               # colonne du nom de fichier dans df
            process_function=convert_rle_to_yolo_labels,
            output_dirs=[Path("dataset/labels")],
            options={
                "df": df_full,
                "label_map": {"catheter": 1, "wire": 2},
                "tol": 0.002,
                "simplify": True,
                "save_params": True,
            },
        )
    """
    output_dir = output_dirs[0]
    image_name = image_path.name  # seul le nom de fichier sert au filtre (comme dans le notebook)

    # --- Filtre du DataFrame sur l'image courante ---
    rows: pd.DataFrame = df[df[filename_col] == image_name]

    if rows.empty:
        logger.warning(f"Aucune annotation trouvée pour '{image_name}' dans le DataFrame. Image ignorée.")
        return None

    # --- Conversion ---
    try:
        converter = YoloSegmentationConverter(image_path)
    except Exception as e:
        logger.error(f"Impossible de charger l'image '{image_path}': {e}")
        return None

    reconstruction_regions = []

    for _, row in rows.iterrows():
        rle_data = row[rle_col]
        brushlabel = row[brushlabels_col]
        class_id = label_map.get(brushlabel, 0)

        try:
            converter.convert(rle_data=rle_data, class_id=class_id, tol=tol, simplify=simplify)
        except Exception as e:
            # On log et on continue : une région défectueuse ne doit pas bloquer les autres
            logger.error(f"Échec conversion région '{brushlabel}' pour '{image_name}': {e}")
            continue

        if save_params:
            reconstruction_regions.append({
                "brushlabels": brushlabel,
                "class_id": class_id,
                "rle": rle_data,
            })

    # Rien n'a été converti (toutes les régions ont échoué)
    if not converter.yolo_lines:
        logger.warning(f"Aucune ligne YOLO générée pour '{image_name}'.")
        return None

    # --- Écriture du fichier de labels YOLO ---
    label_path = output_dir / Path(image_name).with_suffix(".txt")
    converter.write_yolo_lines(label_path)

    # --- Écriture du fichier sidecar de reconstruction ---
    if save_params:
        params = {
            "image_name": image_name,
            "tol": tol,
            "simplify": simplify,
            "regions": reconstruction_regions,
        }
        params_path = output_dir / Path(image_name).with_suffix(".params.json")
        try:
            with params_path.open("w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            logger.info(f"Fichier de reconstruction écrit : {params_path}")
        except (IOError, TypeError) as e:
            # Non bloquant : le label YOLO est déjà écrit
            logger.warning(f"Impossible d'écrire le fichier de reconstruction pour '{image_name}': {e}")

    return label_path