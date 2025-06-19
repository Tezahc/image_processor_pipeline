import cv2
import random
from pathlib import Path
from typing import Any, List, Literal, Optional
from ultralytics.data.utils import IMG_FORMATS
from warnings import warn

ALL_SYMS = ('o', 'h', 'v', 'hv')


def generate_symmetries(
    input_path: Path, 
    output_dirs: List[Path],

    # Options pour contrôler la génération des symétries
    pool: Optional[List[Literal[*ALL_SYMS]]] = None,  # type: ignore
    choose_random: Optional[int] = None,
    include_original: bool = True,
    **options: Any
) -> Optional[List[Path]]:
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
    Optional[List[Path]]
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
    if not output_dirs:
        raise ValueError(f"Erreur [{input_path.name} - Symétrie]: Aucun dossier de sortie ('output_dirs') fourni.")
    output_dir = output_dirs[0]

    if input_path.suffix.lower()[1:] not in IMG_FORMATS:
        # Peut-être ouvrir à tout type d'image ?
        raise ValueError(f"Le fichier {input_path.name} n'est pas un format accepté par Yolo.")

    # Valider le pool si fourni
    pool = pool if pool else list(ALL_SYMS)
    if any(sym not in ALL_SYMS for sym in pool):
        invalid_keys = [key_ for key_ in pool if key_ not in ALL_SYMS]
        raise ValueError(f"`pool` contient des éléments invalides : {invalid_keys}")

    choose_random = len(pool) if choose_random is None else choose_random
    if choose_random > len(pool):
        warn(f"Choix aléatoire de plus d'éléments ({choose_random}) que possible parmi {pool} ({len(pool)}).")
    elif choose_random < 0:
        raise ValueError(f"[{input_path.name} - Symétrie] `choose_random` ({choose_random}) doit être >= 0. Aucune symétrie aléatoire générée.")
    
    # Lire l'image
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"[{input_path.name} - Symétrie] Impossible de charger l'image.")
    
    # Dictionnaire de fonctions génératrices de symétries
    sym_generators = {
        "o" : lambda img: img.copy(),                 # Image originale
        "h" : lambda img: cv2.flip(img, 1),   # Symétrie horizontale 
        "v" : lambda img: cv2.flip(img, 0),   # Symétrie verticale 
        "hv" : lambda img: cv2.flip(img, -1),         # Symétrie h + v (rotation 180°)
    }

    # Sélection des transformations à effectuer
    filter_ = random.sample(pool, choose_random)

    # Ajout de l'original (si non présent)
    if include_original and 'o' not in set(filter_):
        filter_.append('o')

    # Sauvegarde des images générées
    saved_files: List[Path] = []
    for sym in filter_:
        image_flip = sym_generators[sym](image)

        output_filename = input_path.with_stem(f"{input_path.stem}_{sym}")
        output_path = output_dir / output_filename.name

        try:
            success = cv2.imwrite(str(output_path), image_flip)
            if success:
                saved_files.append(output_path)
            else:
                # L'écriture a échoué sans lever d'exception (rare, mais possible)
                # jamais du coup ?
                warn(f"Échec de sauvegarde de la symétrie '{sym}' pour {output_path.name}. Retour False depuis `.imwrite`")
        except Exception as e_save:
            # Erreur lors de l'écriture (permissions, disque plein, etc.)
            warn(f"Erreur [{input_path.name} - Symétrie '{sym}']: Échec de sauvegarde pour {output_filename} : {e_save}.")
            # On continue d'essayer de sauvegarder les autres images
    
    return saved_files
