import cv2
import random
import numpy as np
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple
from warnings import warn


def generate_symmetries(
    file: Path, 
    output_dirs: List[Path],
    choose_random: Optional[int] = None,
    include_original: bool = True,
    pool: Optional[List[Literal['o', 'h', 'v', 'hv']]] = None,
    **options: Any
) -> Optional[Path]:
    # TODO: ajouter et check les vérifications suggérées par gemini (et review...)
    # TODO: ajouter filtre de symétries appliquées
    """
    Génère les symétries d'une image :
    
    - `o` image originale 
    - `h` miroir horizontal 
    - `v` miroir vertical 
    - `hv` miroir horizontal + vertical (rotation 180°)
    
    La fonction utilise OpenCV pour effectuer les flips. FIXME Elle retourne un dictionnaire 
    contenant les différentes versions de l'image.  
    Elle sauvegarde les 4 images résultantes dans le premier dossier de sortie
    fourni (`output_paths[0]`), en ajoutant un suffixe (_o, _h, _v, _hv)
    au nom du fichier original.

    Parameters
    ----------
    file : Path
        Chemin de l'image à traiter. Doit être un fichier PNG valide.
    output_dirs : List[Path]
        Chemin du dossier de sortie. Liste d'un seul élément attendue. 
        *Les éventuels éléments supplémentaires seront ignorés.*
    choose_random : int, optional
        Les symétries générées sont choisies au hasard parmi les options de `pool`. 
        `choose_random` symétries sont créées. Doit être < len(pool)
        Par défaut None
    include_original : bool, optional
        Définit si l'orientation originale doit être inclue systématiquement dans les résultats.  
        Indépendant de `pool`. Ignoré si `choose_random` est `None`  
        - Si False et `pool` inclue 'o'  
            peut quand même produire (au hasard) une image originale dans les résultats.  
        - Si False et `pool` ne contient pas 'o'  
            uniquement les transformations restantes dans pool sont produites et transmises.  
        - Si True et `pool` inclue 'o'  
            BUG peut produire (au hasard) une originale ET une copie dans les résultats.  
        - Si True et `pool` ne contient pas 'o' (désiré)  
            choisi au hasard une orientation parmi pool et ajoute une copie originale.  
        Par défaut True  
    pool : List[Literal['o', 'h', 'v', 'hv']], optional
        Liste des symétries applicables. Si non renseigné, toutes les symétries sont sélectionées
        , par défaut None
    **options : Any
        Accepte des options supplémentaires (ne seront pas utilisées).
    
    Returns
    -------
    FIXME
    dict[str, Path]
        Dictionnaire contenant les chemins des images symétriques générées :
        - "o" : Image originale
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
        raise ValueError(f"Erreur [{file.name} - Symétrie]: Aucun dossier de sortie ('output_dirs') fourni.")
    output_dir = output_dirs[0]

    if file.suffix.lower() != '.png':
        # Peut être ouvrir à tout type d'image ?
        raise ValueError(f"Le fichier {file.name} n'est pas un PNG.")

    # Lire l'image
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image {file.name}.")

    # Génération des symétries (en mémoire)
    symmetries = {
        "o" : image.copy(),         # Image originale
        "h" : cv2.flip(image, 1),   # Symétrie horizontale 
        "v" : cv2.flip(image, 0),   # Symétrie verticale 
        "hv" : cv2.flip(image, -1), # Symétrie h + v (rotation 180°)
    }

    # Sauvegarde des images générés
    saved_files: List[Path] = []
    for suffix, image in symmetries.items():
        output_filename = file.with_stem(f"{file.stem}_{suffix}")
        output_path = output_dir / output_filename.name

        try:
            success = cv2.imwrite(str(output_path), image)
            if success:
                saved_files.append(output_path)
            else:
                # L'écriture a échoué sans lever d'exception (rare mais possible)
                # jamais du coup ?
                warn(f"Échec de sauvegarde de lq symétrie '{suffix}' pour {output_path.name}. Retour False depuis `imwrite`") 
        except Exception as e_save:
            # Erreur lors de l'écriture (permissions, disque plein, etc.)
            print(f"Erreur [{file.name} - Symétrie '{suffix}']: Échec de sauvegarde pour {output_filename} : {e_save}.")
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