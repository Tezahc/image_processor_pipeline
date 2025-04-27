import cv2
import os
import numpy as np

# Chemins des dossiers
input_folder = r"F:\DeepValve\Bionector\Overlays\Videos\OVERLAYS_0150"
output_folder = r"F:\DeepValve\Bionector\Overlays\Videos\OVERLAYS_0150"
os.makedirs(output_folder, exist_ok=True)  # Crée le dossier de sortie s'il n'existe pas

# Parcourir toutes les images du dossier
for image_name in os.listdir(input_folder):
    if image_name.endswith('.png'):  # Filtrer uniquement les fichiers .png
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
        alpha = image[:, :, 3]

        # Trouver les coordonnées des pixels non transparents
        non_transparent_coords = cv2.findNonZero(alpha)

        if non_transparent_coords is None:
            print(f"L'image {image_name} est entièrement transparente, elle sera ignorée.")
            continue

        # Trouver le cadre minimal contenant tous les pixels non transparents
        x, y, w, h = cv2.boundingRect(non_transparent_coords)

        # Recadrer l'image en utilisant ces coordonnées
        cropped_image = image[y:y+h, x:x+w]

        # Sauvegarder l'image recadrée
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, cropped_image)
        print(f"Image recadrée et sauvegardée : {output_path}")

print(f"Traitement terminé. Les images recadrées sont enregistrées dans : {output_folder}")