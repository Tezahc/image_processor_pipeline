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
from ultralytics.data.utils import IMG_FORMATS
try:
    import albumentations as A
except ImportError:
    raise ImportError(
        "Le module 'albumentations' est requis pour cette étape. "
        "Installez-le avec 'pip install albumentations' (ou albumentationsx)."
    )

from ipp.utils import utils
from ipp.utils.artifact import Artifact
from ipp.utils.yolo_labels import YoloLabelHandler


D4Key = Literal["e", "h", "v", "r90", "r180", "r270", "t", "hvt"]
ALL_D4_KEYS = get_args(D4Key)

MODE_POOLS: Dict[str, Tuple[str, ...]] = {
    "symmetry": ("h", "v", "t", "hvt"),          # Les 4 réflexions
    "rotation": ("r90", "r180", "r270"),         # Les 3 rotations non triviales
    "full": ("h", "v", "r90", "r180", "r270", "t", "hvt"),  # D4 sans l'identité
}


def _select_d4_transforms(
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
        Si True, ajoute systématiquement l'image originale ('e') sauf si déjà présente.
    seed : int, optional
        Graine pour la reproductibilité du tirage aléatoire.
    **options : Any
        Options supplémentaires (ignorées).

    Returns
    -------
    Optional[List[Artifact]]
        Un `Artifact` par transformation sauvegardée.
    """
    # 1. Validation
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

    # 2. Chargement de l'image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"[{image_path.name} - D4] Impossible de charger l'image.")
    img_h, img_w = image.shape[:2]

    selected, trace = _select_d4_transforms(mode, pool, choose_random, add_original_copy, seed)

    # 3. Chargement des labels via le handler — une seule lecture pour toutes les transformations.
    #    compose_kwargs / call_kwargs sont identiques pour chaque clé D4 (seul A.D4 change) :
    #    on les construit une fois, avant la boucle.
    compose_kwargs: dict = {}
    call_kwargs: dict = {}
    meta: dict = {}

    if has_labels:
        handler = YoloLabelHandler.from_file(label_path, img_w, img_h)
        compose_kwargs, call_kwargs, meta = handler.to_albumentations(
            img_w, img_h, use_masks=False  # keypoints : D4 est bijectif, aucun point ne sort de l'image
        )

    # 4. Boucle de transformations
    outputs: List[Artifact] = []
    for key in selected:
        # A.D4 doit être instancié par clé (group_element fixé à la construction)
        transform = A.Compose([A.D4(p=1.0, group_element=key)], **compose_kwargs)

        try:
            transformed = transform(image=image, **call_kwargs)
        except Exception as e:
            warn(f"Échec D4 '{key}' sur {image_path.name} : {e}")
            continue

        image_t = transformed["image"]
        img_h_t, img_w_t = image_t.shape[:2]  # r90/r270/t/hvt échangent largeur et hauteur

        image_output_path = utils.build_output_filepath(image_path, image_out_dir, suffix_key=key)
        if not cv2.imwrite(str(image_output_path), image_t):
            warn(f"Échec de sauvegarde de l'image '{key}' pour {image_output_path.name}.")
            continue

        output_entry = Artifact(
            image_path=image_output_path,
            transformation="d4",
            params={"d4_key": key},
            extra={"reproducibility": trace},
        )

        # 5. Reconstruction et enregistrement des labels transformés
        if has_labels:
            label_output_path = utils.build_output_filepath(label_path, label_out_dir, suffix_key=key)

            # Handler vierge aux nouvelles dimensions (r90/r270 échangent w et h)
            out_handler = YoloLabelHandler(img_w_t, img_h_t)

            # Bbox : déjà renormalisées par Albumentations (format 'yolo'), prêtes à l'emploi
            out_handler.update_bboxes(
                transformed.get("bboxes", []),
                transformed.get("bbox_classes", []),
                replace=False,
            )

            # Polygones : keypoints toujours dans le même ordre (remove_invisible=False),
            # poly_lengths permet de regroupe les points par polygone
            out_handler.update_from_keypoints(
                transformed.get("keypoints", []),
                transformed.get("kp_classes", []),
                meta.get("poly_lengths", []),
                img_w_t,
                img_h_t,
                replace=False,
            )

            out_handler.save(label_output_path)
            output_entry.label_path = label_output_path

        outputs.append(output_entry)

    return outputs if outputs else None