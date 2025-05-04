import cv2
import random
import numpy as np
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple
from warnings import warn


SYMS = ('o', 'h', 'v', 'hv')
def generate_symmetries(
    file: Path, 
    output_dirs: List[Path],
    pool: Optional[List[Literal[*SYMS]]] = None, # type: ignore
    choose_random: Optional[int] = None,
    include_original: bool = True,
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
            BUG potentiellement moins d'images qu'attendu => warning "peut être virer le 'o' du pool ?"
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

    pool = pool if pool else list(SYMS)
    # pool = list(SYMS) if not pool else pool ?
    if any(transform not in SYMS for transform in pool):
        raise ValueError("`pool` contient des éléments interdits")

    if choose_random and choose_random >= len(pool):
        warn(f"Choix aléatoire de plus d'éléments ({choose_random}) que possible parmis {pool} ({len(pool)}).")
    
    # Lire l'image
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image {file.name}.")
    
    # Dictionnaire de fonctions génératrices de symétries
    sym_generators = {
        "o" : lambda img: img.copy(),         # Image originale
        "h" : lambda img: cv2.flip(img, 1),   # Symétrie horizontale 
        "v" : lambda img: cv2.flip(img, 0),   # Symétrie verticale 
        "hv" : lambda img: cv2.flip(img, -1), # Symétrie h + v (rotation 180°)
    }

    symmetries = {}
    random.shuffle(pool)
    # HACK: zip permet de s'arrêter soit dès que la pool est finie soit dès que le nombre de pick rng est atteint
    for sym, rng in zip(pool, range(choose_random)):
        # Application de la symétrie
        symmetries[sym] = sym_generators[sym](image)
    
    # Ajout de l'original (peut déjà être présent par le for)
    if include_original and 'o' not in set(symmetries.keys()):
        symmetries['o'] = sym_generators["o"](image)

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
