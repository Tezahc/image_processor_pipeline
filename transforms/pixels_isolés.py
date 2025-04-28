from pathlib import Path
import cv2
import numpy as np


def keep_biggest_chunk(file: Path, min_component_size = 500):
    if file.suffix.lower() != '.png':
        raise ValueError(f"Le fichier {file.name} n'est pas un PNG.")
    
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image {file.name}.")

    # Vérifier si l'image contient un canal alpha
    if image.shape[2] != 4:
        raise AttributeError(f"L'image {file.name} ne contient pas de canal alpha, elle sera ignorée.")
    # TODO: ignorer réellement les erreurs afin de passer les fichiers problématiques. Peut être garder une trace dans l'orchestrateur

    # Extraire le canal alpha
    b, g, r, alpha = cv2.split(image)

    # Binariser le canal alpha (255 = opaque, 0 = transparent)
    _, binary_alpha = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    # Détecter les composants connectés
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_alpha, connectivity=8)

    # Trouver l'index du plus grand composant (supposé être l'objet principal)
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):  # On ignore le fond (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i

    # Créer un masque conservant uniquement l'objet principal
    clean_mask = np.where(labels == max_label, 255, 0).astype(np.uint8)

    # Supprimer les petits objets parasites
    for i in range(1, num_labels):
        if i != max_label and stats[i, cv2.CC_STAT_AREA] < min_component_size:
            clean_mask[labels == i] = 0  # Supprime ces zones

    # Appliquer le masque pour rendre transparent les composants isolés
    alpha[clean_mask == 0] = 0

    # Recomposer l'image avec le canal alpha nettoyé
    cleaned_image = cv2.merge((b, g, r, alpha))

    return cleaned_image
