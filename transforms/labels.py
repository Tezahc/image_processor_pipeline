from pathlib import Path
from typing import Any, List, Optional
from icecream import ic


def change_label_class(input_path: Path,
                       output_dirs: List[Path],
                       # **options
                       new_label_class_id: int = 0,
                       **options: Any
                       ) -> Optional[Path]:
    """Change l'id de classe d'un fichier label (format yolo)

    Parameters
    ----------
    input_path : Path
        Chemin du fichier label à modifier.  
        Ne fonctionne qu'avec un format YOLO : "class_id x_min y_min x_max y_max"
    output_dirs : List[Path]
        Liste (un seul élément attendu) du chemin de dossier d'enregistrement.  
        NOTE : actuellement totalement ignoré, écrase le fichier d'entrée
    new_label_class_id : int, optional
        id de la classe à remplacer, by default 0

    TODO : ajouter un filtre pour ne remplacer que cette classe d'origine
    TODO : gérer multiples bbox (lignes). (actuellement, seulement la première ligne)

    Returns
    -------
    Optional[Path]
        Renvoie le chemin du fichier modifié. None s'il y a eu un problème.
    """

    # Vérifications : 
    # if not ouput_dirs => devrait être géré par l'orchestrateur
    ouput_dir = output_dirs[0]

    try:
        with input_path.open('r+', encoding='utf-8') as f:
            line = f.readline()
            f.seek(0)
            f.truncate()
            line = line.split()
            line[0] = str(new_label_class_id)
            f.write(' '.join(line))
        return input_path
    except Exception as e:
        print(f"Problème : {e}")
        return None

if __name__ == '__main__':
    #tests
    file = Path(r"C:\Users\GuillaumeChazet\Documents\ICUREsearch\DeepValve\MicroClave_001_h_r047.txt")
    output = [Path.cwd()]

    change_label_class(file, output, 12)