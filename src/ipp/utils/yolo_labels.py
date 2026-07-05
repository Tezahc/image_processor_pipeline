"""
ipp/utils/yolo_labels.py

Gestion unifiée des annotations YOLO (bounding box et segmentation polygonale),
mélangées ou non au sein d'un même fichier `.txt`.

Objectif : centraliser en un seul endroit la lecture, l'écriture et les
conversions vers les formats attendus par Albumentations (bboxes, keypoints,
masks), pour éviter que chaque fonction de traitement (crop, rotation,
symétrie, D4, ...) ne réinvente sa propre logique de parsing/conversion.

Organisation
------------
- `YoloLabel` (ABC) : une annotation = une ligne de fichier `.txt`.
    - `BBoxLabel`         : bbox classique [cx, cy, w, h] normalisée.
    - `SegmentationLabel` : polygone [x1, y1, ..., xn, yn] normalisé.
- `YoloLabelHandler` : collection de `YoloLabel` pour une image donnée.
    Gère la lecture/écriture fichier et présente les labels "à la demande"
    dans le format voulu (bbox / keypoints / masks), en délégant la
    conversion individuelle à chaque `YoloLabel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn


# ---------------------------------------------------------------------------
#  Labels individuels (une instance = une ligne de fichier YOLO)
# ---------------------------------------------------------------------------

class YoloLabel(ABC):
    """Représente une annotation YOLO unique.

    Classe abstraite commune aux bbox et aux polygones de segmentation.
    Ne porte que ce qui est réellement partagé : l'identifiant de classe,
    un score de confiance optionnel (prédictions), et la capacité à se
    sérialiser en une ligne de texte YOLO.
    """

    kind: ClassVar[str]  # "bbox" ou "seg" — utile pour le debug/repr

    def __init__(self, class_id: int, confidence: Optional[float] = None):
        self.class_id = class_id
        self.confidence = confidence

    @staticmethod
    def from_line(line: str) -> "YoloLabel":
        """Factory : instancie `BBoxLabel` ou `SegmentationLabel` selon le
        nombre de coordonnées trouvées sur la ligne.

        - 4 coordonnées  -> bbox
        - 5 coordonnées  -> bbox + score de confiance (prédiction)
        - pair >= 6      -> polygone de segmentation

        Raises
        ------
        ValueError
            Si le nombre de coordonnées ne correspond à aucun format connu.
        """
        parts = line.strip().split()
        if not parts:
            raise ValueError("Ligne vide.")

        class_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        if len(coords) == 4:
            return BBoxLabel(class_id, np.array(coords, dtype=np.float32))
        elif len(coords) == 5:
            return BBoxLabel(class_id, np.array(coords[:4], dtype=np.float32), confidence=coords[4])
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            return SegmentationLabel(class_id, points)
        else:
            raise ValueError(
                f"Format non reconnu ({len(coords)} coordonnées) sur la ligne : '{line.strip()}'"
            )

    @abstractmethod
    def to_line(self) -> str:
        """Sérialise l'annotation en une ligne de texte au format YOLO (normalisé)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(class_id={self.class_id})"


class BBoxLabel(YoloLabel):
    """Bounding box YOLO classique, stockée au format canon xywh normalisé."""

    kind = "bbox"

    def __init__(self, class_id: int, xywhn: np.ndarray, confidence: Optional[float] = None):
        super().__init__(class_id, confidence)
        self.xywhn = np.asarray(xywhn, dtype=np.float32).reshape(4)

    @classmethod
    def from_xyxy(cls, class_id: int, xyxy: np.ndarray, img_w: int, img_h: int,
                  **kw) -> "BBoxLabel":
        """Construit une BBoxLabel depuis des coordonnées pixels absolues [x1,y1,x2,y2]."""
        xywhn = xyxy2xywhn(np.array([xyxy], dtype=np.float32), w=img_w, h=img_h).squeeze(0)
        return cls(class_id, xywhn, **kw)

    def to_xyxy(self, img_w: int, img_h: int) -> np.ndarray:
        """Retourne la bbox en pixels absolus [x1, y1, x2, y2]."""
        return xywhn2xyxy(np.array([self.xywhn]), w=img_w, h=img_h).squeeze(0)

    def to_line(self) -> str:
        cx, cy, w, h = self.xywhn
        base = f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        return f"{base} {self.confidence:.6f}" if self.confidence is not None else base


class SegmentationLabel(YoloLabel):
    """Polygone de segmentation YOLO, stocké au format canon points normalisés (N, 2)."""

    kind = "seg"

    def __init__(self, class_id: int, points_n: np.ndarray, confidence: Optional[float] = None):
        super().__init__(class_id, confidence)
        self.points_n = np.asarray(points_n, dtype=np.float32).reshape(-1, 2)

    @property
    def n_points(self) -> int:
        return self.points_n.shape[0]

    # --- Conversions individuelles (à la demande, une seule à la fois) ---

    def to_keypoints(self, img_w: int, img_h: int) -> List[Tuple[float, float]]:
        """Dénormalise les points en pixels absolus, format attendu par
        `A.KeypointParams(format='xy')`."""
        abs_points = self.points_n * [img_w, img_h]
        return [tuple(p) for p in abs_points]

    def to_mask(self, img_w: int, img_h: int) -> np.ndarray:
        """Rasterise le polygone en masque binaire (0/255), une instance = un masque."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        abs_points = (self.points_n * [img_w, img_h]).astype(np.int32)
        cv2.fillPoly(mask, [abs_points], 255)
        return mask

    @classmethod
    def from_keypoints(cls, class_id: int, keypoints: List[Tuple[float, float]],
                        img_w: int, img_h: int, **kw) -> "SegmentationLabel":
        """Reconstruit un label depuis des keypoints absolus (après transformation)."""
        pts = np.array(keypoints, dtype=np.float32)
        pts_n = np.clip(pts / [img_w, img_h], 0.0, 1.0)
        return cls(class_id, pts_n, **kw)

    @classmethod
    def from_mask(cls, class_id: int, mask: np.ndarray, **kw) -> List["SegmentationLabel"]:
        """Extrait les contours d'un masque transformé et retourne un label par
        contour valide (un masque peut contenir plusieurs polygones disjoints
        après une transformation géométrique)."""
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        labels = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            pts = contour.reshape(-1, 2).astype(np.float32)
            pts_n = pts / [w, h]
            labels.append(cls(class_id, pts_n, **kw))
        return labels

    def to_line(self) -> str:
        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points_n)
        return f"{self.class_id} {coords_str}"


# ---------------------------------------------------------------------------
#  Handler : collection de labels pour une image + I/O + vues Albumentations
# ---------------------------------------------------------------------------

class YoloLabelHandler:
    """Gère l'ensemble des labels (bbox et/ou segmentation, mélangés) d'une image.

    Responsabilités :
    - lecture / écriture du fichier `.txt` YOLO ;
    - stockage de la collection sous forme d'une simple liste de `YoloLabel` ;
    - présentation "à la demande" dans les formats attendus par Albumentations
      (bboxes, keypoints, masks), en délégant la conversion élément par
      élément à chaque `YoloLabel` ;
    - reconstruction de la collection à partir des sorties d'Albumentations.

    `img_w`/`img_h` sont optionnels à l'instanciation : ils ne sont
    nécessaires qu'au moment des conversions impliquant des pixels absolus
    (`as_keypoints`, `as_masks`), et peuvent alors être fournis directement
    à l'appel de la méthode plutôt qu'à la construction du handler.
    """

    def __init__(self, img_w: Optional[int] = None, img_h: Optional[int] = None):
        self.width = img_w
        self.height = img_h
        self.labels: List[YoloLabel] = []

    # --- Lecture / écriture ---

    @classmethod
    def from_file(cls, label_path: str | Path,
                  img_w: Optional[int] = None, img_h: Optional[int] = None) -> "YoloLabelHandler":
        handler = cls(img_w, img_h)
        handler.load(label_path)
        return handler

    def load(self, label_path: str | Path) -> None:
        """Charge (ajoute) les labels depuis un fichier `.txt` YOLO.

        N'écrase pas les labels déjà présents dans le handler (permet de
        fusionner plusieurs sources). Un fichier absent n'est pas une
        erreur : cela signifie simplement une image "négative" (sans
        annotation).
        """
        label_path = Path(label_path)
        if not label_path.exists():
            return
        with label_path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    self.labels.append(YoloLabel.from_line(line))
                except ValueError as e:
                    raise ValueError(f"{label_path.name}, ligne {line_num} : {e}") from e

    def add(self, label: YoloLabel) -> None:
        self.labels.append(label)

    def save(self, out_path: str | Path) -> None:
        out_path = Path(out_path)
        with out_path.open("w", encoding="utf-8") as f:
            for label in self.labels:
                f.write(label.to_line() + "\n")

    # --- Filtres pratiques ---

    @property
    def bbox_labels(self) -> List[BBoxLabel]:
        return [l for l in self.labels if isinstance(l, BBoxLabel)]

    @property
    def seg_labels(self) -> List[SegmentationLabel]:
        return [l for l in self.labels if isinstance(l, SegmentationLabel)]

    # --- Présentation pour Albumentations (lecture) ---

    def as_bboxes(self) -> Tuple[List[list], List[int]]:
        """(bboxes, classes) au format attendu par `A.BboxParams(format='yolo')`."""
        labels = self.bbox_labels
        return [l.xywhn.tolist() for l in labels], [l.class_id for l in labels]

    def as_keypoints(self, img_w: Optional[int] = None, img_h: Optional[int] = None
                      ) -> Tuple[List[Tuple[float, float]], List[int], List[int]]:
        """Aplatit tous les polygones en une liste unique de keypoints absolus.

        Retourne aussi une classe "à plat" par point (requis par
        Albumentations) et `poly_lengths` (nb de points par polygone),
        indispensable pour regrouper les keypoints après transformation.
        """
        img_w, img_h = self._resolve_dims(img_w, img_h)
        labels = self.seg_labels
        keypoints, flat_classes, poly_lengths = [], [], []
        for l in labels:
            kps = l.to_keypoints(img_w, img_h)
            keypoints.extend(kps)
            flat_classes.extend([l.class_id] * len(kps))
            poly_lengths.append(len(kps))
        return keypoints, flat_classes, poly_lengths

    def as_masks(self, img_w: Optional[int] = None, img_h: Optional[int] = None
                 ) -> Tuple[List[np.ndarray], List[int]]:
        """Rasterise chaque polygone en masque binaire individuel (une instance
        = un masque), pour les pipelines Albumentations basés sur `masks=`."""
        img_w, img_h = self._resolve_dims(img_w, img_h)
        labels = self.seg_labels
        return [l.to_mask(img_w, img_h) for l in labels], [l.class_id for l in labels]

    def to_albumentations(self, img_w: Optional[int] = None, img_h: Optional[int] = None,
                           use_masks: bool = False) -> Tuple[dict, dict, dict]:
        """Bonus : construit les kwargs prêts à l'emploi pour Albumentations,
        en n'ajoutant `bbox_params`/`keypoint_params` que si des labels du
        type correspondant existent (évite les erreurs Albumentations avec
        des listes vides — logique jusqu'ici dupliquée dans chaque fonction).

        Returns
        -------
        compose_kwargs : dict à passer à `A.Compose(transforms, **compose_kwargs)`
        call_kwargs    : dict à passer à `transform(image=img, **call_kwargs)`
        meta           : infos nécessaires pour reconstruire après transfo
                         (`poly_lengths` si `use_masks=False`)
        """
        import albumentations as A  # import local : pas de dépendance dure au module

        compose_kwargs: dict = {}
        call_kwargs: dict = {}
        meta: dict = {}

        bboxes, bbox_classes = self.as_bboxes()
        if bboxes:
            compose_kwargs["bbox_params"] = A.BboxParams(format="yolo", label_fields=["bbox_classes"])
            call_kwargs["bboxes"] = bboxes
            call_kwargs["bbox_classes"] = bbox_classes

        if self.seg_labels:
            if use_masks:
                masks, seg_classes = self.as_masks(img_w, img_h)
                call_kwargs["masks"] = masks
                meta["seg_classes"] = seg_classes
            else:
                keypoints, flat_classes, poly_lengths = self.as_keypoints(img_w, img_h)
                compose_kwargs["keypoint_params"] = A.KeypointParams(
                    format="xy", label_fields=["kp_classes"], remove_invisible=False
                )
                call_kwargs["keypoints"] = keypoints
                call_kwargs["kp_classes"] = flat_classes
                meta["poly_lengths"] = poly_lengths

        return compose_kwargs, call_kwargs, meta

    def _resolve_dims(self, img_w: Optional[int], img_h: Optional[int]) -> Tuple[int, int]:
        img_w = img_w or self.width
        img_h = img_h or self.height
        if img_w is None or img_h is None:
            raise ValueError(
                "Dimensions de l'image inconnues : fournir img_w/img_h à l'appel "
                "ou les définir à l'instanciation du handler."
            )
        return img_w, img_h

    # --- Reconstruction depuis les sorties Albumentations (écriture) ---

    def update_from_keypoints(self, keypoints: List[Tuple[float, float]], flat_classes: List[int],
                               poly_lengths: List[int], img_w: int, img_h: int,
                               replace: bool = True) -> None:
        """Regroupe des keypoints transformés (via `poly_lengths`) et les
        réinjecte comme `SegmentationLabel`, normalisés.

        Si `replace`, les anciennes segmentations sont retirées avant ajout
        (les bbox du handler restent inchangées).
        """
        if replace:
            self.labels = self.bbox_labels
        idx = 0
        for length in poly_lengths:
            cls_id = flat_classes[idx]
            pts = keypoints[idx: idx + length]
            idx += length
            self.labels.append(SegmentationLabel.from_keypoints(cls_id, pts, img_w, img_h))

    def update_from_masks(self, masks: List[np.ndarray], classes: List[int],
                           replace: bool = True) -> None:
        """Reconstruit les `SegmentationLabel` depuis des masques transformés
        (un masque par instance, même ordre que `as_masks`)."""
        if replace:
            self.labels = self.bbox_labels
        for mask, cls_id in zip(masks, classes):
            self.labels.extend(SegmentationLabel.from_mask(cls_id, mask))

    def update_bboxes(self, bboxes: List[list], classes: List[int], replace: bool = True) -> None:
        """Réinjecte des bboxes transformées (déjà normalisées, format yolo)."""
        if replace:
            self.labels = self.seg_labels
        for bbox, cls_id in zip(bboxes, classes):
            self.labels.append(BBoxLabel(cls_id, np.asarray(bbox, dtype=np.float32)))

    # --- Divers ---

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __repr__(self) -> str:
        return f"YoloLabelHandler({len(self.bbox_labels)} bbox, {len(self.seg_labels)} seg)"
