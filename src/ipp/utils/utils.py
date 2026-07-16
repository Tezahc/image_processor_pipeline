from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def check_path(folder_name, root=None):
    """
    Construit un chemin complet pour un dossier en fonction de son nom et d'un chemin racine optionnel.

    Cette fonction vérifie si le nom de dossier fourni est un chemin absolu ou relatif.
    Si c'est un chemin absolu, il est retourné tel quel. Sinon, il est combiné avec le chemin racine
    spécifié ou le répertoire de travail actuel si aucun chemin racine n'est fourni.

    :param folder_name: Le nom du dossier ou le chemin du dossier à vérifier.
                        Peut être un chemin relatif ou absolu.
    :type folder_name: str ou Path
    :param root: Le chemin racine à utiliser si folder_name est un chemin relatif.
                 Si non spécifié, le répertoire de travail actuel est utilisé.
                 Ignoré si folder_name est un chemin absolu
    :type root: str ou Path, optional
    :return: Le chemin complet du dossier.
    :rtype: Path
    """
    # Convertir folder_name en objet Path
    path = Path(folder_name)

    # Déterminer le chemin racine : utiliser root s'il est fourni, sinon utiliser le répertoire de travail actuel
    root_path = Path(root) if root else Path('.')

    # Vérifier si le chemin est absolu
    if path.is_absolute():
        # Retourner le chemin absolu tel quel
        return path
    else:
        # Combiner le chemin relatif avec le chemin racine
        return root_path / path

def _validate_dirs(output_dirs: List[Path], nb_dirs: int) -> Path | Tuple[Path, ...]:
    """Vérifie que le bon nombre de répertoires de sortie sont fournis.

    Parameters
    ----------
    output_dirs : List[Path]
        Liste des répertoires de sortie.
    nb_dirs : int
        Nombre de répertoires attendus à vérifier

    Returns
    -------
    Tuple[Path]
        Tuple des répertoires fournis.
    
    Raises
    ------
    IndexError
        Si moins de `nb_dirs` dossiers sont fournis.
    """
    if len(output_dirs) < nb_dirs:
        raise IndexError(f"Au moins {nb_dirs} dossiers de sortie requis (images, labels). {len(output_dirs)} fournis.")
    elif len(output_dirs) > nb_dirs:
        #TODO: warn
        print(f"WARNING : plus de dossiers de sorties que nécessaires. Seul les {nb_dirs} seront utilisés")
    
    paths = tuple(Path(dir_) for dir_ in output_dirs)
    if nb_dirs == 1:
        return paths[0]
    return paths

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

def _load_image(filepath: Path) -> np.ndarray:
    """Charge une image avec OpenCV.

    Parameters
    ----------
    filepath : Path
        Chemin vers l'image.

    Returns
    -------
    np.ndarray
        Image BGR.
    
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    IOError
        Si OpenCV n'arrive pas à lire l'image.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Image non trouvée: {filepath}")
    img = cv2.imread(str(filepath))
    if img is None:
        raise IOError(f"Impossible de charger l'image {filepath.name} via OpenCV.")
    return img

def load_image_pil(image_path: Path) -> Image:
    """Charge une image du dataset à partir de son nom et renvoie l'image, sa largeur et sa hauteur.
    
    L'orientation de l'image est automatiquement corrigée en se basant sur les données EXIF.
    
    Args:
        image_name (str | Path): Le nom du fichier de l'image à charger.

    Returns:
        Tuple[Image.Image, int, int]: Un tuple contenant l'objet image PIL,
                                      sa largeur (w) et sa hauteur (h).
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image non trouvée: {image_path}")
    image = Image.open(image_path)
    ImageOps.exif_transpose(image, in_place=True)
    w, h = image.size
    
    return image, w, h

def _read_bboxes(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Lit un fichier de labels YOLO (.txt) et renvoie les classes et bboxes.

    Parameters
    ----------
    filepath : Path
        Chemin du fichier `.txt`

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - classes: shape (N, 1), dtype=int
        - bboxes: shape (N, 4), format [cx, cy, w, h] normalisés
    
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le contenu est invalide.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Fichier label non trouvé : {filepath}")
    data = np.loadtxt(filepath, ndmin=2)
    try:
        classes = data[:, 0].astype(int)
        bboxes = data[:, 1:5].astype(float)
    except Exception as e:
        raise ValueError(f"Format invalide dans {filepath.name}: {e}")
    return classes, bboxes

def _save_yolo_labels(
    label_path: Path,
    classes: np.ndarray,
    bboxes: np.ndarray
) -> None:
    """Save yolo labels to a text file"""
    data = np.column_stack((classes, bboxes))
    np.savetxt(label_path, data, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])


def build_output_filepath(
    input_file: Path,
    output_dir: Path,
    *,
    suffix_key: Optional[str] = None,
    idx: Optional[int] = None,
    name_format: str = "{key}{idx:03d}",
    separator: str = "_",
    **_
) -> Path:
    """
    Build an output file path from an input file path, optionally adding
    a suffix to the base filename.

    The suffix is appended to the input filename stem, separated by
    ``separator``. The suffix behavior can be controlled via ``suffix_key``
    and ``idx``.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the input file.
    output_dir : pathlib.Path
        Path to the output file folder.
    suffix_key : str or None, optional
        Suffix identifier to append to the filename stem.

        - ``None`` or ``"off"`` disables suffix addition.
        - Any non-empty string is used as a suffix.
        - An empty string is invalid and raises ``ValueError``.

    idx : int or None, optional
        Optional index used when generating the suffix. If provided, the
        suffix is formatted using ``name_format``; otherwise, ``suffix_key``
        is appended as-is.

    name_format : str, optional
        Format string used to build the suffix when ``idx`` is provided.
        The format string must accept the fields ``key`` and ``idx``.

        Example: ``"{key}{idx:03d}"`` → ``r005``

    separator : str, optional
        Separator used between the original filename stem and the suffix.

    **_ :
        Additional keyword arguments are ignored. This allows transparent
        forwarding of keyword arguments from higher-level processing
        functions.

    Returns
    -------
    pathlib.Path
        The constructed output file path.

    Raises
    ------
    ValueError
        If ``suffix_key`` is an empty string.
    TypeError
        If ``suffix_key`` is not a string when provided.

    Examples
    --------
    >>> build_output_filepath(Path("file.csv"), Path("out"))
    PosixPath('out/file.csv')

    >>> build_output_filepath(Path("file.csv"), Path("out"), suffix_key="r")
    PosixPath('out/file_r.csv')

    >>> build_output_filepath(
    ...     Path("file.csv"), Path("out"), suffix_key="r", idx=2
    ... )
    PosixPath('out/file_r002.csv')

    >>> build_output_filepath(
    ...     Path("file.csv"), Path("out"), suffix_key="off"
    ... )
    PosixPath('out/file.csv')
    """

    stem = input_file.stem
    #TODO: permettre de changer l'extension en argument, sinon rOmet celle d'origine
    file_suffix = input_file.suffix

    parts = [stem]

    if suffix_key not in (None, "off"):
        if not isinstance(suffix_key, str) or not suffix_key:
            raise ValueError("suffixe de nom de fichier invalide")
        
        # mise en forme avec l'index si présent
        suffix = (name_format.format(key=suffix_key, idx=idx) 
                  if idx is not None 
                  else suffix_key)
        parts.append(suffix)

    new_stem = separator.join(parts)

    return output_dir / f"{new_stem}{file_suffix}"
