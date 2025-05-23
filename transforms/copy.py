import shutil
from pathlib import Path
from typing import Any, List, Optional


def copy_files(
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