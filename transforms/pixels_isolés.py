import cv2
import numpy as np
import os

# Chemins des dossiers
input_folder = r"F:\DeepValve\Bionector\Overlays\Videos\OVERLAYS_0150"
output_folder = r"F:\DeepValve\Bionector\Overlays\Videos\OVERLAYS_0150"
os.makedirs(output_folder, exist_ok=True)  # Crée le dossier de sortie s'il n'existe pas

# Seuil minimal pour garder un composant (évite de supprimer l'objet principal)
min_component_size = 500  # Ajustez selon vos images

# Parcourir toutes les images du dossier
for image_name in os.listdir(input_folder):
    if image_name.endswith('.png'):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Erreur : Impossible de charger l'image {image_name}.")
            continue

        # Vérifier si l'image contient un canal alpha
        if image.shape[2] != 4:
            print(f"L'image {image_name} ne contient pas de canal alpha, elle sera ignorée.")
            continue

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

        # Sauvegarder l'image nettoyée
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, cleaned_image)
        print(f"Image nettoyée et sauvegardée : {output_path}")

print(f"Traitement terminé. Les images nettoyées sont enregistrées dans : {output_folder}")