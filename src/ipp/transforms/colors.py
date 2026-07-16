"""
ipp/transforms/brightness_contrast.py

Ajustement aléatoire de luminosité et de contraste via AlbumentationsX.
La transformation est purement photométrique : aucune géométrie n'est modifiée,
les labels ne sont donc ni lus ni écrits.
"""

import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from warnings import warn

import cv2

try:
    import albumentations as A
except ImportError:
    raise ImportError(
        "AlbumentationsX est requis. Installer avec 'uv add albumentationsx'."
    )

from ipp.utils.artifact import Artifact
from ipp.utils import utils


def apply_brightness_contrast(
    *inputs: Path,
    output_dirs: List[Path],
    brightness_limit: Union[float, Tuple[float, float]] = 0.2,
    contrast_limit: Union[float, Tuple[float, float]] = 0.2,
    p: float = 0.8,
    seed: Optional[int] = None,
    **options: Any,
) -> Optional[Artifact]:
    """
    Applique un ajustement aléatoire de luminosité et de contraste à une image.

    La transformation étant purement photométrique, les labels ne sont ni attendus
    ni traités. Les éventuels chemins de labels passés en `inputs` supplémentaires
    sont silencieusement ignorés.

    Parameters
    ----------
    *inputs : Path
        `inputs[0]` = image à traiter. Les éléments suivants sont ignorés.
    output_dirs : List[Path]
        Un seul dossier de sortie (images).
    brightness_limit : float or (float, float), default=0.2
        Plage d'ajustement de la luminosité.
        - Scalaire `v`  → plage symétrique `[-v, +v]`.
        - Tuple `(a, b)` → plage `[a, b]`.
    contrast_limit : float or (float, float), default=0.2
        Plage d'ajustement du contraste. Même convention que `brightness_limit`.
    p : float, default=0.8
        Probabilité d'application de la transformation pour cet appel.
        Avec une seed par image (comportement par défaut), chaque image a une
        chance indépendante de `p` d'être transformée.
        Si `p < 1.0` et que le tirage est défavorable, la fonction retourne None
        (aucun fichier de sortie créé — cf. section Notes).
    seed : int, optional
        Seed pour la reproductibilité. Si None (défaut), une seed est générée
        aléatoirement par appel et stockée dans l'Artifact.
        Cf. section Notes pour le comportement selon le contexte d'appel.
    **options : Any
        Transmis à `build_output_filepath` (suffix_key, idx, separator, ...).

    Returns
    -------
    Optional[Artifact]
        - `Artifact` contenant le chemin de sortie et les paramètres exacts
          appliqués (brightness et contrast effectivement échantillonnés).
        - `None` si `p < 1.0` et que la transformation n'a pas été déclenchée
          pour cet appel.

    Notes
    -----
    **Comportement de la seed**

    La seed initialise le générateur aléatoire interne d'Albumentations pour cet
    appel. Elle contrôle deux décisions successives dans l'ordre :
    1. Le tirage de probabilité pour `p` (apply ou skip).
    2. L'échantillonnage des valeurs de brightness et contrast.

    *Même seed sur deux images différentes* : les valeurs de brightness et contrast
    appliquées seront **identiques** (même état RNG → même séquence de tirages),
    mais le résultat visuel sera différent car les pixels sources diffèrent.

    *Pour un pipeline avec des paramètres indépendants par image* (comportement
    recommandé) : ne pas passer de seed, ou dériver une seed par image depuis une
    seed globale ::

        global_rng = random.Random(global_seed)

        # Option A — seeds séquentielles (ordre-dépendant, pleinement reproductible)
        options = {"seed": global_rng.randint(0, 2**32 - 1)}

        # Option B — seed depuis le nom de fichier (reproductible sans stocker l'ordre)
        options = {"seed": hash(image_path.name) % 2**32}

    *Pour reproduire exactement une transformation passée* : deux méthodes
    équivalentes, la seconde étant plus robuste (ne dépend pas du RNG Albumentations) :

        # Méthode 1 : re-passer la même seed (re-joue toute la séquence RNG)
        apply_brightness_contrast(image_path, seed=artifact.params["seed"], ...)

        # Méthode 2 : forcer les valeurs exactes échantillonnées (plus robuste)
        apply_brightness_contrast(
            image_path,
            brightness_limit=(v, v),   # v = artifact.params["brightness"]
            contrast_limit=(v, v),     # v = artifact.params["contrast"]
            p=1.0,                     # garantit l'application
            ...
        )

    **Subset via `p`**

    Avec une seed **par image** : chaque image a une chance indépendante de `p`
    d'être transformée → fraction transformée ≈ `p` sur l'ensemble du dataset.

    Avec une seed **globale identique pour toutes les images** : toutes les images
    reçoivent la même décision (toutes transformées ou toutes skippées) — ce n'est
    généralement pas l'effet recherché.

    **`save_applied_params` et Artifact**

    `save_applied_params=True` sur la transform individuelle demande à AlbumentationsX
    de retourner les valeurs exactement échantillonnées (brightness=+0.12, pas
    seulement brightness_limit=0.2). Ces valeurs sont plus précieuses que la seed
    seule car elles permettent de reconstruire la transformation sans dépendre du
    mécanisme RNG interne d'Albumentations. Les deux sont stockés dans l'Artifact.
    """
    image_path = inputs[0]

    if len(inputs) > 1:
        warn(
            f"[brightness_contrast] {len(inputs) - 1} input(s) supplémentaire(s) ignoré(s) "
            f"(labels inutiles pour une transformation photométrique)."
        )

    image_out_dir = utils._validate_dirs(output_dirs, 1)

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    # Seed : génération aléatoire si absente, stockage systématique dans l'Artifact
    # pour permettre la reconstruction ultérieure même sans save_applied_params.
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    transform = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_range=(-brightness_limit, brightness_limit),
                contrast_range=(-contrast_limit, contrast_limit),
                p=p,
            )
        ],
        save_applied_params=True,  # capture brightness et contrast exacts
        seed=seed,
    )

    transformed = transform(image=image)

    # Récupération des paramètres effectivement appliqués.
    # AlbumentationsX place les params sous la clé du nom de la classe dans
    # transformed["applied_transforms"]. Structure attendue :
    #   [("RandomBrightnessContrast", {"brightness": 0.12, "contrast": -0.08})]
    # ou absente / vide si la transformation a été skippée (tirage p défavorable).

    applied_transforms: dict = transformed.get("applied_transforms", [])
    was_applied = False
    bc_params = {}

    if applied_transforms and isinstance(applied_transforms, list):
        for item in applied_transforms:
            if isinstance(item, tuple) and len(item) == 2:
                transform_name, params_dict = item
                if "RandomBrightnessContrast" not in str(transform_name):
                    continue
                was_applied=True
                bc_params = params_dict
                break

    output_image = transformed["image"] if was_applied else image

    out_image_path = utils.build_output_filepath(image_path, image_out_dir, **options)
    if not cv2.imwrite(str(out_image_path), output_image):
        raise IOError(f"Échec écriture image : {out_image_path}")
    
    # Extraire les valeurs appliquées (ou 0 si skippée)
    brightness = bc_params.get("brightness", bc_params.get("brightness_range", 0.0)) or 0.0
    contrast = bc_params.get("contrast", bc_params.get("contrast_range", 0.0)) or 0.0
    
    return [Artifact(
        image_path=out_image_path,
        transformation=apply_brightness_contrast.__name__,
        params={
            # --- Reproductibilité par seed (méthode 1) ---
            "seed": seed,
            # --- Reproductibilité par valeurs exactes (méthode 2, plus robuste) ---
            # Ces valeurs permettent de reconstruire sans dépendre du RNG interne
            "brightness": brightness,
            "contrast": contrast,
            "was_applied":was_applied
        },
    )]