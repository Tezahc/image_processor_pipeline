from pathlib import Path
import cv2
import numpy as np

def crop_to_alpha_bounding_box(file: Path) -> np.ndarray:
    """
    Recadre une image PNG en utilisant son canal alpha pour détecter la zone utile.
    
    Cette fonction charge une image avec un canal alpha, détecte tous les pixels non-transparents, 
    calcule la plus petite boîte englobante autour de ces pixels, puis extrait et retourne la sous-image.
    
    Parameters
    ----------
    file : Path
        Chemin vers le fichier image à traiter. Doit être un fichier PNG valide avec un canal alpha.

    Returns
    -------
    np.ndarray
        L'image recadrée sous forme de tableau NumPy.

    Raises
    ------
    ValueError
        Si le fichier n'est pas un PNG, ne contient pas de canal alpha, ou est entièrement transparent.
    FileNotFoundError
        Si l'image ne peut pas être chargée.
    """
    # Vérifier l'extension du fichier
    if file.suffix.lower() != '.png':
        raise ValueError(f"Le fichier {file.name} n'est pas un PNG.")
    
    # Charger l'image avec tous les canaux
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image {file.name}.")

    # Vérifier que l'image possède bien 4 canaux (dont alpha)
    if image.shape[2] != 4:
        raise ValueError(f"L'image {file.name} ne possède pas de canal alpha.")

    # Extraire le canal alpha
    alpha = image[:, :, 3]

    # Trouver les coordonnées des pixels non-transparents
    non_transparent_coords = cv2.findNonZero(alpha)

    if non_transparent_coords is None:
        raise ValueError(f"L'image {file.name} est entièrement transparente.")

    # Calculer la plus petite boîte englobante
    x, y, w, h = cv2.boundingRect(non_transparent_coords)

    # Recadrer l'image selon cette boîte
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image
