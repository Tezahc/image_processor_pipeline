"""
Génération de transformations D4 (rotations 90° + symétries) pour des paires
image / label YOLO (bbox classique et/ou segmentation polygonale).

Cette version s'appuie directement sur la bibliothèque Albumentations (et son 
augmentation A.D4) pour gérer toute la logique géométrique de manière robuste 
sur l'image et ses annotations.

Les formats de labels (BBox vs Segmentation) sont détectés automatiquement 
ligne par ligne. Les polygones de segmentation sont traduits en `keypoints` 
absolus pour Albumentations, puis renormalisés à l'enregistrement.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args
from warnings import warn

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    raise ImportError(
        "Le module 'albumentations' est requis pour cette étape. "
        "Installez-le avec 'pip install albumentations' (ou albumentationsx)."
    )

from ultralytics.data.utils import IMG_FORMATS

from ipp.utils.artifact import Artifact
from ipp.utils import utils

D4Key = Literal["e", "h", "v", "r90", "r180", "r270", "t", "hvt"]
ALL_D4_KEYS = get_args(D4Key)

MODE_POOLS: Dict[str, Tuple[str, ...]] = {
    "symmetry": ("h", "v", "t", "hvt"),          # Les 4 réflexions
    "rotation": ("r90", "r180", "r270"),         # Les 3 rotations non triviales
    "full": ("h", "v", "r90", "r180", "r270", "t", "hvt"),  # D4 sans l'identité
}


def select_d4_transforms(
    mode: str,
    pool: Optional[List[str]],
    choose_random: Optional[int],
    add_original_copy: bool,
    seed: Optional[int],
) -> Tuple[List[D4Key], Dict[str, Any]]:
    """Résout la liste des transformations D4 à appliquer et construit la trace
    de reproductibilité associée.
    """
    if mode == "custom":
        if not pool:
            raise ValueError("`pool` doit être une liste non vide pour le mode 'custom'.")
        base_pool = list(pool)
    elif mode in MODE_POOLS:
        if pool is not None:
            warn(f"`pool` est ignoré : le mode '{mode}' définit déjà son propre pool "
                 "(utiliser mode='custom' pour fournir un pool personnalisé).")
        base_pool = list(MODE_POOLS[mode])
    else:
        raise ValueError(f"Mode '{mode}' invalide. Choisir parmi {list(MODE_POOLS) + ['custom']}")

    invalid = [k for k in base_pool if k not in ALL_D4_KEYS]
    if invalid:
        raise ValueError(f"Clés D4 invalides dans le pool : {invalid}. Attendu parmi {ALL_D4_KEYS}")

    rng = random.Random(seed) if seed is not None else random

    if choose_random is None:
        selected = list(base_pool)
    else:
        if choose_random < 0:
            raise ValueError("`choose_random` doit être positif.")
        if choose_random > len(base_pool):
            warn(f"`choose_random` ({choose_random}) > taille du pool ({len(base_pool)}). "
                 "Toutes les transformations du pool seront utilisées.")
            choose_random = len(base_pool)
        selected = rng.sample(base_pool, choose_random)

    if add_original_copy and "e" not in selected:
        selected = ["e"] + selected

    trace = {
        "mode": mode,
        "pool_candidates": base_pool,
        "choose_random": choose_random,
        "seed": seed,
        "add_original_copy": add_original_copy,
        "selected": selected,
    }
    return selected, trace


def _read_yolo_labels(
    label_path: Path, 
    img_w: int, 
    img_h: int
) -> Tuple[List[list], List[int], List[tuple], List[int], List[int]]:
    """Lit un fichier de labels YOLO et sépare les BBox des Polygones.
    Détecte automatiquement le format selon le nombre de coordonnées.
    Dénormalise les polygones en pixels absolus pour Albumentations (keypoints).
    """
    bboxes = []
    bbox_classes = []
    
    keypoints = []
    poly_classes = []
    poly_lengths = []
    
    with label_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
            except ValueError as e:
                raise ValueError(f"Ligne {line_num} invalide dans {label_path.name} : {e}") from e
            
            if len(coords) == 4:
                # BBox classique [cx, cy, w, h] normalisée
                bboxes.append(coords)
                bbox_classes.append(cls_id)
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # Segmentation [x1, y1, x2, y2...] normalisée
                poly_lengths.append(len(coords) // 2)
                poly_classes.append(cls_id)
                for i in range(0, len(coords), 2):
                    # Dénormalisation en pixels absolus pour A.KeypointParams(format='xy')
                    x = coords[i] * img_w
                    y = coords[i+1] * img_h
                    keypoints.append((x, y))
            else:
                #TODO: implem en cas d'impair : le dernier item = score confiance
                raise ValueError(
                    f"Ligne {line_num} de {label_path.name} : format non reconnu "
                    f"({len(coords)} coordonnées). Attendu 4 (bbox) ou pair >= 6 (polygone)."
                )
                
    return bboxes, bbox_classes, keypoints, poly_classes, poly_lengths


def _write_yolo_labels(
    out_path: Path,
    bboxes: List[list],
    bbox_classes: List[int],
    keypoints: List[tuple],
    poly_classes: List[int],
    poly_lengths: List[int],
    img_w: int,
    img_h: int
) -> None:
    """Enregistre les BBoxes et les Polygones dans un fichier YOLO.
    Renormalise les polygones en [0, 1] en fonction des dimensions (transformées) de l'image.
    """
    with out_path.open("w", encoding="utf-8") as f:
        # 1. BBoxes (déjà normalisées par Albumentations, format='yolo')
        for cls_id, bbox in zip(bbox_classes, bboxes):
            coords_str = " ".join(f"{x:.6f}" for x in bbox)
            f.write(f"{cls_id} {coords_str}\n")
            
        # 2. Polygones (à renormaliser depuis les keypoints)
        kp_idx = 0
        for cls_id, length in zip(poly_classes, poly_lengths):
            poly_kps = keypoints[kp_idx : kp_idx + length]
            kp_idx += length
            
            poly_coords = []
            for x, y in poly_kps:
                # Renormalisation [0, 1] par rapport aux nouvelles dimensions (img_w, img_h)
                nx = max(0.0, min(1.0, x / img_w))
                ny = max(0.0, min(1.0, y / img_h))
                poly_coords.extend([nx, ny])
                
            coords_str = " ".join(f"{v:.6f}" for v in poly_coords)
            f.write(f"{cls_id} {coords_str}\n")


def generate_d4_transforms(
    *inputs: Path,
    output_dirs: List[Path],
    mode: Literal["symmetry", "rotation", "full", "custom"] = "full",
    pool: Optional[List[str]] = None,
    choose_random: Optional[int] = None,
    add_original_copy: bool = True,
    seed: Optional[int] = None,
    **options: Any,
) -> Optional[List[Artifact]]:
    """
    Génère les transformations D4 (symétries + rotations 90°) d'une image en
    s'appuyant sur Albumentations, avec gestion unifiée de labels YOLO 
    (bbox classiques et segmentation polygonale).
    La détection du format (bbox vs segmentation) est automatique ligne par ligne.

    Repose sur `A.D4` d'Albumentations : le groupe diédral D4 comprend
    l'identité, 3 rotations (90°, 180°, 270°) et 4 réflexions (horizontale,
    verticale, et les 2 diagonales). Voir le docstring du module pour le
    détail des 8 clés.

    Contrairement à `generate_symmetries`, l'identité ('e') n'est jamais un
    membre sélectionnable du pool : elle est uniquement ajoutée via
    `add_original_copy`, ce qui évite l'ambiguïté/bug de l'ancienne fonction
    (double comptage possible de l'original selon la présence de 'o' dans le
    pool ET `include_original`).

    Parameters
    ----------
    inputs : Path
        1 ou 2 chemins : `(image_path,)` ou `(image_path, label_path)`.
    output_dirs : List[Path]
        `output_dirs[0]` : dossier des images. `output_dirs[1]` (optionnel) :
        dossier des labels, requis pour traiter les annotations.
    mode : {'symmetry', 'rotation', 'full', 'custom'}, default='full'
        Détermine le pool de transformations candidates.
        'custom' utilise le `pool` fourni.
    pool : list[str], optional
        Pool explicite de clés D4 parmi {'e', 'h', 'v', 'r90', 'r180', 'r270', 't', 'hvt'}.
        Requis si `mode='custom'` ; ignoré sinon.
    choose_random : int, optional
        Tire aléatoirement ce nombre de transformations uniques dans le pool.
        Si `None`, applique tout le pool.
    add_original_copy : bool, default=True
        Si True, ajoute systématiquement l'image originale ('e') sauf si déjà présente
    seed : int, optional
        Graine pour la reproductibilité du tirage aléatoire.
    **options : Any
        Options supplémentaires (ignorées).

    Returns
    -------
    Optional[List[Artifact]]
        Un `Artifact` par transformation sauvegardée.
    """
    # 1. Validation des arguments et dossiers
    if len(inputs) == 1:
        image_path, label_path = inputs[0], None
    elif len(inputs) == 2:
        image_path, label_path = inputs
    else:
        raise ValueError(f"`generate_d4_transforms` attend 1 ou 2 inputs, reçu {len(inputs)}")

    has_labels = label_path is not None
    
    if not output_dirs:
        raise ValueError(f"Erreur [{image_path.name} - D4] : aucun dossier de sortie ('output_dirs') fourni.")
    if has_labels and len(output_dirs) < 2:
        raise ValueError(
            f"Erreur [{image_path.name} - D4] : des labels sont fournis mais un seul dossier "
            "de sortie est configuré. Deux dossiers (images, labels) sont requis."
        )

    image_out_dir = output_dirs[0]
    label_out_dir = output_dirs[1] if has_labels else None

    if image_path.suffix.lower()[1:] not in IMG_FORMATS:
        raise ValueError(f"Le fichier {image_path.name} n'est pas un format d'image supporté.")

    # 2. Chargement des données
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"[{image_path.name} - D4] Impossible de charger l'image.")

    selected, trace = select_d4_transforms(mode, pool, choose_random, add_original_copy, seed)

    classes = bboxes = keypoints = poly_classes = poly_lengths = None
    kp_classes = []
    
    if has_labels:
        img_h, img_w = image.shape[:2]
        bboxes, bbox_classes, keypoints, poly_classes, poly_lengths = _read_yolo_labels(
            label_path, img_w, img_h
        )
        # Création d'une liste de classes "plate" pour que chaque keypoint ait son label
        # (requis par le fonctionnement strict d'Albumentations pour les keypoints)
        for c, length in zip(poly_classes, poly_lengths):
            kp_classes.extend([c] * length)

    # 3. Application des transformations via Albumentations
    outputs: List[Artifact] = []
    for key in selected:
        compose_kwargs = {}
        call_kwargs = {"image": image}
        
        # Configuration dynamique d'Albumentations.
        # N'ajoute les paramètres (BBox, Keypoints) QUE s'ils sont présents dans le label
        # pour éviter des erreurs inutiles d'Albumentations avec des arrays vides.
        if has_labels:
            if bboxes:
                compose_kwargs["bbox_params"] = A.BboxParams(format="yolo", label_fields=["bbox_classes"])
                call_kwargs["bboxes"] = bboxes
                call_kwargs["bbox_classes"] = bbox_classes
            if keypoints:
                compose_kwargs["keypoint_params"] = A.KeypointParams(
                    coord_format="xy", label_fields=["kp_classes"], remove_invisible=False
                )
                call_kwargs["keypoints"] = keypoints
                call_kwargs["kp_classes"] = kp_classes

        # Instanciation de la transformation "en force" sur la déclinaison voulue via group_element
        transform = A.Compose([A.D4(p=1.0, group_element=key)], **compose_kwargs)
        
        try:
            transformed = transform(**call_kwargs)
        except Exception as e:
            warn(f"Échec de l'application de la transformation D4 '{key}' sur {image_path.name} : {e}")
            continue
            
        image_t = transformed["image"]
        image_output_path = utils.build_output_filepath(image_path, image_out_dir, suffix_key=key)
        
        success = cv2.imwrite(str(image_output_path), image_t)
        if not success:
            warn(f"Échec de sauvegarde de l'image transformée '{key}' pour {image_output_path.name}.")
            continue
            
        output_entry = Artifact(
            image_path=image_output_path,
            transformation="d4",
            params={"d4_key": key},
            extra={"reproducibility": trace},
        )
        
        # Enregistrement des labels
        if has_labels:
            bboxes_t = transformed.get("bboxes", [])
            bbox_classes_t = transformed.get("bbox_classes", [])
            keypoints_t = transformed.get("keypoints", [])
            
            label_output_path = utils.build_output_filepath(label_path, label_out_dir, suffix_key=key)
            img_h_t, img_w_t = image_t.shape[:2]  # Nouvelles dimensions de l'image transformée
            
            _write_yolo_labels(
                label_output_path,
                bboxes_t,
                bbox_classes_t,
                keypoints_t,
                poly_classes,  
                poly_lengths,
                img_w_t,
                img_h_t,
            )
            output_entry.label_path = label_output_path
            
        outputs.append(output_entry)

    if not outputs:
        return None

    return outputs