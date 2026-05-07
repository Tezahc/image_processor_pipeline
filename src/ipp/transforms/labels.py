from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional


def change_label_class(input_path: Path,
                       output_dirs: List[Path],
                       # **options
                       cls_mapping: Dict[int, int] = {0:0},
                       **options: Any
                       ) -> Optional[Path]:
    """
    Modifie les ID de classe d'un fichier label (format yolo) en utilisant
    un dictionnaire de mapping et sauvegarde le résultat dans un nouveau dossier.

    Cette fonction lit un fichier de labels ligne par ligne. Pour chaque ligne,
    elle vérifie si l'ID de classe est une clé dans le dictionnaire de mapping.
    Si c'est le cas, elle le remplace par la valeur correspondante.
    Sinon, l'ID de classe reste inchangé.

    Parameters
    ----------
    input_path : Path
        Chemin du fichier label à modifier.
        Format YOLO attendu : "class_id x_center y_center width height"
    output_dirs : List[Path]
        Liste contenant le chemin du dossier d'enregistrement. 
        Seul le premier élément est utilisé.
    class_id_mapping : Dict[int, int]
        Dictionnaire de mapping des anciens ID de classe (clés) vers les
        nouveaux ID de classe (valeurs). Ex: {0: 99, 2: 50}
    options : Any
        Autres options (actuellement non utilisées).

    Returns
    -------
    Optional[Path]
        Renvoie le chemin du nouveau fichier modifié.
        Renvoie None en cas d'erreur.
    """

    # Vérifications : 
    # if not ouput_dirs => devrait être géré par l'orchestrateur
    output_dir = output_dirs[0]
    output_path = output_dir / input_path.name

    try:
        with input_path.open('r', encoding='utf-8') as in_file, output_path.open('w', encoding='utf-8') as out_file:
            for line in in_file:
                parts = line.strip().split()
                # Si la ligne est vide
                if not parts:
                    continue
                current_cls_id = int(parts[0])
                new_cls_id = cls_mapping.get(current_cls_id, current_cls_id)
                parts[0] = str(new_cls_id)
                new_line = ' '.join(parts)
                out_file.write(f"{new_line}\n")
        return output_path
    except Exception as e:
        print(f"Problème : {e}")
        # suppression du nouveau fichier
        if output_path.exists(): output_path.unlink()
        return None

if __name__ == '__main__':
    # --- Section de test avec des fichiers temporaires ---
    
    test_dir = Path(tempfile.mkdtemp())
    input_dir = test_dir / "input_labels"
    output_dir = test_dir / "output_labels"
    input_dir.mkdir()
    
    # Créer un fichier de test avec diverses classes
    test_file_path = input_dir / "test_label.txt"
    test_content = (
        "0 0.5 0.5 0.1 0.1\n"  # Classe 0
        "1 0.2 0.2 0.1 0.1\n"  # Classe 1
        "0 0.8 0.8 0.1 0.1\n"  # Classe 0
        "2 0.3 0.3 0.1 0.1\n"  # Classe 2 (ne sera pas dans le mapping)
    )
    test_file_path.write_text(test_content, encoding='utf-8')

    print(f"Fichier de test créé : {test_file_path}")
    print("--- Contenu original ---")
    print(test_content)
    
    # --- Cas de test : Changer la classe 0 en 99 et la classe 1 en 77 ---
    # La classe 2 doit rester inchangée car elle n'est pas dans le mapping.
    print("\n--- Test: Changer classe 0 -> 99 et 1 -> 77 ---")
    
    # Définition du dictionnaire de mapping
    mapping_to_apply = {
        0: 99,
        1: 77
    }
    
    output_path = change_label_class(
        input_path=test_file_path,
        output_dirs=[output_dir],
        class_id_mapping=mapping_to_apply
    )

    if output_path and output_path.exists():
        print(f"Fichier de sortie généré : {output_path}")
        print("--- Contenu du fichier de sortie ---")
        output_content = output_path.read_text(encoding='utf-8')
        print(output_content)
        
        print("--- Vérification ---")
        expected_content = (
            "99 0.5 0.5 0.1 0.1\n"
            "77 0.2 0.2 0.1 0.1\n"
            "99 0.8 0.8 0.1 0.1\n"
            "2 0.3 0.3 0.1 0.1\n"
        )
        if output_content == expected_content:
            print("Succès : Le contenu du fichier est correct.")
        else:
            print("Échec : Le contenu du fichier est incorrect.")
            print("Attendu :\n" + expected_content)
    else:
        print("Le test a échoué.")

    # Nettoyer les fichiers et dossiers de test
    print(f"\nNettoyage du dossier de test : {test_dir}")
    shutil.rmtree(test_dir)