from pathlib import Path
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np
import random



def generate_symmetries(
    file: Path, 
    output_dirs: List[Path],
    **options: Any
) -> Optional[Path]:
    # TODO: ajouter et check les vérifications suggérées par gemini (et review...)
    """
    Génère les symétries d'une image : imagee, miroir horizontal, miroir vertical,
    miroir horizontal + vertical (rotation 180°).
    
    La fonction utilise OpenCV pour effectuer les flips. Elle retourne un dictionnaire 
    contenant les différentes versions de l'image.

    Parameters
    ----------
    file : Path
        Chemin du fichier image à traiter. Doit être un fichier PNG valide.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionnaire contenant les images symétriques :
        - "o" : Image imagee
        - "h" : Symétrie horizontale (flip gauche-droite)
        - "v" : Symétrie verticale (flip haut-bas)
        - "hv": Symétrie horizontale + verticale (rotation 180°)

    Raises
    ------
    ValueError
        Si le fichier n'est pas un PNG.
    FileNotFoundError
        Si l'image ne peut pas être chargée.
    """
    if not output_dirs:
        print(f"Erreur [{file.name} - Symétrie Aléatoire]: Aucun dossier de sortie ('output_dirs') fourni.")
        return None
    output_dir = output_dirs[0]

    if file.suffix.lower() != '.png':
        raise ValueError(f"Le fichier {file.name} n'est pas un PNG.")

    # Lire l'image
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image {file.name}.")

    # génération des symétries (en mémoire)
    symetries = {
        "o" : image.copy(),         # Image originale
        "h" : cv2.flip(image, 1),   # Symétrie horizontale 
        "v" : cv2.flip(image, 0),   # Symétrie verticale 
        "hv" : cv2.flip(image, -1), # Symétrie horizontale + verticale (équivalent à une rotation de 180°)
    }

    saved_files: List[Path] = []
    for key, image in symetries.items():
        output_filename = f"{file.stem}_{key}{file.suffix}"
        output_path = output_dir / output_filename

        try:
            success = cv2.imwrite(str(output_path), image)
            if success:
                saved_files.append(output_path)
            else:
                # L'écriture a échoué sans lever d'exception (rare mais possible)
                print(f"Avertissement [{file.name} - Symétrie]: Échec de sauvegarde (imwrite a retourné False) pour {output_filename}")
        except Exception as e_save:
            # Erreur lors de l'écriture (permissions, disque plein, etc.)
            print(f"Erreur [{file.name} - Symétrie]: Échec de sauvegarde pour {output_filename}: {e_save}")
            # On continue d'essayer de sauvegarder les autres images
    
    return saved_files


def generate_random_symmetry(
    input_path: Path,
    output_dirs: List[Path],
    **options: Any
) -> Optional[Path]: # Retourne un seul Path ou None
    """
    Génère et SAUVEGARDE une symétrie aléatoire d'une image.

    Choisit aléatoirement parmi : originale, symétrie horizontale (H),
    symétrie verticale (V), ou H+V (rotation 180°). Sauvegarde
    l'image résultante dans le premier dossier de sortie (`output_paths[0]`)
    en ajoutant le suffixe correspondant (_o, _h, _v, _hv) au nom original.

    Args:
        input_path (Path): Chemin du fichier image à traiter.
        output_paths (List[Path]): Liste des chemins des dossiers de sortie configurés.
                                   Le premier dossier (output_paths[0]) sera utilisé.
        **options (Any): Accepte des options supplémentaires (non utilisées ici).

    Returns:
        Optional[Path]:
            - Path: Le chemin complet vers le fichier unique sauvegardé.
            - None: Si l'image d'entrée ne peut pas être lue, si aucun dossier de sortie
                    n'est fourni, ou si la sauvegarde échoue.
    """
    # --- 1. Vérifications Préliminaires ---
    if not output_dirs:
        print(f"Erreur [{input_path.name} - Symétrie Aléatoire]: Aucun dossier de sortie ('output_dir') fourni.")
        return None
    output_dir = output_dirs[0]

    # --- 2. Lecture de l'image d'entrée ---
    try:
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Erreur [{input_path.name} - Symétrie Aléatoire]: Impossible de charger l'image.")
            return None
    except Exception as e:
        print(f"Erreur [{input_path.name} - Symétrie Aléatoire]: Échec lors de la lecture du fichier: {e}")
        return None

    # --- 3. Choix aléatoire de l'orientation ---
    # pas de calcul ici car lambda !
    orientations = {
        "o": lambda img: img.copy(),        # Originale
        "h": lambda img: cv2.flip(img, 1),  # Horizontale
        "v": lambda img: cv2.flip(img, 0),  # Verticale
        "hv": lambda img: cv2.flip(img, -1) # H+V
    }
    chosen_key = random.choice(list(orientations.keys()))
    chosen_function = orientations[chosen_key]

    # --- 4. Génération de l'image choisie ---
    try:
        output_image = chosen_function(image)
        # print(f"Info [{input_path.name} - Symétrie Aléatoire]: Orientation choisie: '{chosen_key}'")
    except Exception as e:
         print(f"Erreur [{input_path.name} - Symétrie Aléatoire]: Échec lors de la génération du flip '{chosen_key}': {e}")
         return None

    # --- 5. Sauvegarde de l'image unique ---
    output_filename = f"{input_path.stem}_{chosen_key}{input_path.suffix}"
    output_file_path = output_dir / output_filename

    try:
        success = cv2.imwrite(str(output_file_path), output_image)

        if success:
            # --- 6. Retourner le chemin unique sauvegardé ---
            return output_file_path
        else:
            print(f"Avertissement [{input_path.name} - Symétrie Aléatoire]: Échec de sauvegarde (imwrite a retourné False) pour {output_filename}")
            return None
    except Exception as e_save:
        print(f"Erreur [{input_path.name} - Symétrie Aléatoire]: Échec de sauvegarde pour {output_filename}: {e_save}")
        return None