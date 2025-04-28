from pathlib import Path
import cv2
import numpy as np

def generate_symmetries(file: Path) -> dict[str, np.ndarray]:
    """
    Génère les symétries d'une image : originale, miroir horizontal, miroir vertical,
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
    if file.suffix.lower() != '.png':
        raise ValueError(f"Le fichier {file.name} n'est pas un PNG.")

    # Lire l'image
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image {file.name}.")

    # Image originale
    original = image.copy()

    # Symétrie horizontale (gauche-droite)
    h_flip = cv2.flip(original, 1)

    # Symétrie verticale (haut-bas)
    v_flip = cv2.flip(original, 0)

    # Symétrie horizontale + verticale (équivalent à une rotation de 180°)
    hv_flip = cv2.flip(original, -1)

    return {
        "o": original,
        "h": h_flip,
        "v": v_flip,
        "hv": hv_flip
    }
