import cv2
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
import numpy as np
from ultralytics.data.utils import IMG_FORMATS
from warnings import warn
from ipp.utils.artifact import Artifact
from ipp.utils import utils


SymmetryKey = Literal['o', 'h', 'v', 'hv']
ALL_SYMS :Tuple[SymmetryKey, ...]= ('o', 'h', 'v', 'hv')

class SymmetrySpec(TypedDict):
    key: SymmetryKey
    flip_x: bool
    flip_y: bool

SYMMETRIES: dict[SymmetryKey, SymmetrySpec] = {
    "o": {"key": "o", "flip_x": False, "flip_y": False},
    "h": {"key": "h", "flip_x": True, "flip_y": False},
    "v": {"key": "v", "flip_x": False, "flip_y": True},
    "hv": {"key": "hv", "flip_x": True, "flip_y": True}
}

def select_symmetries(
    pool: list[SymmetryKey],
    choose_random: int | None,
    include_original: bool
) -> list[SymmetrySpec]:
    """Select and return symmetry specifications."""
    pool = pool if pool is not None else list(ALL_SYMS)

    if any(sym not in ALL_SYMS for sym in pool):
        invalid = [s for s in pool if s not in ALL_SYMS]
        raise ValueError(f"`pool` contient une clé de symétrie invalide: {invalid}. Attendu {ALL_SYMS}")
    
    if choose_random is None:
        selected = list(pool)
    else:
        if choose_random < 0:
            raise ValueError(f"`choose_random` doit être positif")
        if choose_random > len(pool):
            warn(f"`choose_random` ({choose_random}) > pool size ({len(pool)})."
                 f"Pioche dans toutes les symétries disponibles")
            choose_random = len(pool)
        selected = random.sample(pool, choose_random)
    
    if include_original and "o" not in selected:
        selected.appen("o")
    
    return [SYMMETRIES[key] for key in selected]


def _read_yolo_label(label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read yolo labels from a text file"""
    data = np.loadtxt(label_path, ndmin=2)
    classes = data[:, 0].astype(int)
    bboxes = data[:, 1:].astype(float)
    return classes, bboxes

def _save_yolo_labels(
    label_path: Path,
    classes: np.ndarray,
    bboxes: np.ndarray
) -> None:
    """Save yolo labels to a text file"""
    data = np.column_stack((classes, bboxes))
    np.savetxt(label_path, data, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])

def apply_symmetry_image(img: np.ndarray, spec: SymmetrySpec) -> np.ndarray:
    """Apply symmetry to an image."""
    if not spec["flip_x"] and not spec["flip_y"]:
        return img.copy()
    if spec["flip_x"] and spec["flip_y"]:
        return cv2.flip(img, -1)
    if spec["flip_x"]:
        return cv2.flip(img, 1)
    if spec["flip_y"]:
        return cv2.flip(img, 0)
    
    # non atteignable (normalement...)
    return img.copy()

def apply_symmetry_bboxes(
    bboxes: np.ndarray,
    spec: SymmetrySpec
) -> np.ndarray:
    """
    Apply a symmetry transformation to YOLO-normalized bounding boxes.

    Parameters
    ----------
    bboxes : np.ndarray
        Array of shape (N, 4) in YOLO format (x_center, y_center, w, h),
        normalized in [0, 1].
    spec : SymmetrySpec
        Symmetry specification.

    Returns
    -------
    np.ndarray
        Transformed bounding boxes, same shape and format.
    """
    bboxes_out = bboxes.copy()

    if spec["flip_x"]: 
        bboxes_out[:, 0] = 1.0 - bboxes_out[:, 0]
    if spec["flip_y"]: 
        bboxes_out[:, 1] = 1.0 - bboxes_out[:, 1]

    return bboxes_out


def generate_symmetries(
    *inputs: Path,  # permet d'adapter avec ou sans label
    output_dirs: List[Path],

    # Options pour contrôler la génération des symétries
    pool: Optional[List[SymmetryKey]] = None, 
    choose_random: Optional[int] = None,
    include_original: bool = True,
    **options: Any
) -> Optional[list[Artifact]]:
    """
    Génère les symétries d'une image :
    
    - `o` image originale 
    - `h` miroir horizontal 
    - `v` miroir vertical 
    - `hv` miroir horizontal + vertical (rotation 180°)
    
    Permet soit de générer toutes les symétries spécifiées dans `pool`,
    soit de choisir aléatoirement un nombre défini (`choose_random`)
    d'orientations uniques à partir de `pool`. L'original peut être inclus
    forcément via `include_original`.
    La fonction utilise OpenCV pour effectuer les flips. 
    Elle sauvegarde les 4 images résultantes dans le premier dossier de sortie
    fourni (`output_paths[0]`), en ajoutant un suffixe (_o, _h, _v, _hv)
    au nom du fichier original.

    Parameters
    ----------
    input_path : Path
        Chemin de l'image à traiter. Doit être un fichier PNG valide.
    output_dirs : List[Path]
        Chemin du dossier de sortie. Liste d'un seul élément attendue. 
        *Les éventuels éléments supplémentaires seront ignorés.*
    pool : List[Literal['o', 'h', 'v', 'hv']], optional
        Liste des symétries applicables. Si non renseigné, toutes les symétries sont sélectionnées
        , par défaut None
    choose_random : int, optional
        Choisit aléatoirement ce nombre d'orientations *uniques* dans le `pool`. 
        Si omis, génère *toutes* les orientations du `pool`.
        Par défaut None
    include_original : bool, optional
        Si True, assure que l'orientation originale ('o')
        est incluse dans les sorties, même si non présente dans le pool ou
        non choisie aléatoirement.

        Définit si l'orientation originale doit être incluse systématiquement dans les résultats.  
        Indépendant de `pool`. Ignoré si `choose_random` est `None`  
        - Si False et `pool` inclue 'o'  
            peut quand même produire (au hasard) une image originale dans les résultats.  
        - Si False et `pool` ne contient pas 'o'  
            uniquement les transformations restantes dans pool sont produites et transmises.  
        - Si True et `pool` inclue 'o'  
            BUG : potentiellement moins d'images qu'attendu => warning "peut être virer le 'o' du pool ?"
        - Si True et `pool` ne contient pas 'o' (désiré)  
            choisi au hasard une orientation parmi pool et ajoute une copie originale.  
        Par défaut True  
    **options : Any
        Options supplémentaires (ignorées).
    
    Returns
    -------
    Optional[list[Artifact]]
        Liste des chemins des fichiers sauvegardés, ou None si
        une erreur initiale se produit ou si aucune sauvegarde ne réussit.

    Raises
    ------
    ValueError
        Si le fichier n'est pas un PNG,
        si `output_paths` est vide, 
        si `pool` contient des clés invalides,
        si `choose_random` est > nombre d'éléments dans `pool` (après filtrage potentiel de 'o').
        ou si `choose_random` est < 0
    FileNotFoundError
        Si l'image ne peut pas être chargée par OpenCV.
    """
    # adapte si label est fourni ou non
    if len(inputs) == 1:
        image_path = inputs[0]
        label_path = None
    elif len(inputs) == 2:
        image_path, label_path = inputs
    else:
        raise ValueError(f"`generate_symmetries` attend 1 ou 2 inputs, reçu {len(inputs)}")
    
    if not output_dirs:
        raise ValueError(f"Erreur [{image_path.name} - Symétrie]: Aucun dossier de sortie ('output_dirs') fourni.")
    image_out_dir = output_dirs[0]

    if image_path.suffix.lower()[1:] not in IMG_FORMATS:
        # Peut-être ouvrir à tout type d'image ?
        raise ValueError(f"Le fichier {image_path.name} n'est pas un format accepté par Yolo.")

    # Lire l'image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"[{image_path.name} - Symétrie] Impossible de charger l'image.")
    
    # Crée le pool de symétries à faire
    specs = select_symmetries(pool, choose_random, include_original)

    # Sauvegarde des images générées
    outputs: List[Artifact] = []
    for spec in specs:
        # --- Image ---
        image_flip = apply_symmetry_image(image, spec)
        image_output_path = utils.build_output_filepath(image_path, image_out_dir, suffix_key=spec["key"])

        success = cv2.imwrite(str(image_output_path), image_flip)
        if not success:
            warn(f"Échec de sauvegarde de la symétrie '{spec['key']}' pour {image_output_path.name}. "
                 "Retour False depuis `.imwrite`")
            continue

        # --- Output entry (image toujours présente) ---
        output_entry = Artifact(
            image_path = image_output_path,
            transformation = "symetry",
            params = {
                "symmetry": spec["key"],
                "flip_x": spec["flip_x"],
                "flip_y": spec["flip_y"]
            }
        )

        # --- Labels (optionnels) ---
        if label_path is not None and len(output_dirs) > 1:
            classes, bboxes = _read_yolo_label(label_path)
            bboxes_sym = apply_symmetry_bboxes(bboxes, spec)

            #TODO: check la présence du dossier d'output des labels plus proprement
            label_output_path = utils.build_output_filepath(label_path, output_dirs[1], suffix_key=spec["key"])
            _save_yolo_labels(label_output_path, classes, bboxes_sym)

            output_entry.label_path = label_output_path
        outputs.append(output_entry)

    if not outputs:
        return None

    return outputs
