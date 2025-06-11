import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple


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