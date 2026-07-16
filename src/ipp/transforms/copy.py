from pathlib import Path
import shutil
from typing import Any, List, Optional, Tuple

from PIL import Image, ImageOps
from ipp.utils.artifact import Artifact
from ipp.utils import utils


def copy_img_with_labels(
    input_image_path: Path, 
    input_label_path: Path,
    output_dirs: List[Path],
    **options: Any
) -> Optional[List[Path]]:
    """Copie des fichiers vers un nouveau dossier"""

    if len(output_dirs) < 2:
        raise ValueError(f"Pas assez de dossiers de sortie. {output_dirs}")
    output_image = output_dirs[0]
    output_label = output_dirs[1]

    try:
        img_out = shutil.copy2(input_image_path, output_image)
        lbl_out = shutil.copy2(input_label_path, output_label)
        return [Path(img_out), Path(lbl_out)]
    
    except IOError as io:
        print(f"Impossible de copier le fichier : {io}")
        return None
    except Exception as e:
        print(f"Autre erreur : {e}")
        return None
    
def copy_files(
    input_file: Path,
    output_dirs: List[Path],
    suffix: str=None,
    replace_params: Tuple[str, str]=None,
    **options: Any
) -> Optional[List[Path]]:
    """copie des fichiers depuis 1 dossier vers un autre
    
    Args:
        input_file (Path): fichier à copier.
        output_dirs (List[Path]): dossier de destination (un seul attendu).
        suffix (str): suffixe à ajouter à la fin du nom de fichier de destination.
        replace_params (Tuple[str, str]): Paramètres passés à la fonction `file.name.replace(arg0, arg1)` pour le nom du fichier de destination.

    Returns:
        Optionnal[Path]

    """
    output_dir = utils._validate_dirs(output_dirs, 1)

    if suffix and replace_params:
        raise ValueError(f"un seul des 2 paramètres `replace_param` et `suffix` doit être renseigné")
    
    # Copie en fonction des paramètres. 
    # Suffixe ajoute le paramètre à la fin du nom. Séparé par un underscore `_`
    if suffix:
        output_name = output_dir / input_file.with_stem(f"{input_file.stem}_{suffix}").name
        out = shutil.copy2(input_file, output_name)
    # Remplace une partie du nom
    elif replace_params:
        output_name = output_dir / input_file.name.replace(replace_params[0], replace_params[1])
        out = shutil.copy2(input_file, output_name)
    else:
        out = shutil.copy2(input_file, output_dir)
    
    return Path(out)

def copy_from_label(
    input_label: Path,
    output_dirs: List[Path],
    images_source_dir: Path,
    **options: Any
) -> Optional[List[Artifact]]:
    output_dir = utils._validate_dirs(output_dirs, 1)

    try:
        image_path = next(images_source_dir.rglob(f"{input_label.stem}*"))
    except StopIteration:
        raise StopIteration(f"fichier image {input_label.stem} non trouvé.")
    
    image = Image.open(image_path)
    ImageOps.exif_transpose(image, in_place=True)
    image.save(output_dir/image_path.name)

    artifact = Artifact(input_label, 
                        "copy_from_label", 
                        params={"images_dir": images_source_dir, "image_path": image_path})
    
    return [artifact]

def copy_images(
    input_image: Path,
    output_dirs: List[Path],
    **options: Any
) -> Optional[List[Artifact]]:
    output_dir = utils._validate_dirs(output_dirs, 1)

    image, w, h = utils.load_image_pil(input_image)

    image_path_out = utils.build_output_filepath(input_image, output_dir, suffix_key=options.get("suffix") or None)
    image.save(image_path_out)

    return [Artifact(image_path_out, copy_images.__name__, {})]