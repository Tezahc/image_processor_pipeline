import cv2
import numpy as np
import os

# Chemin des images
input_folder = r"F:\DeepValve\Bionector\Overlays\Videos\IMAGES_0141"
output_folder = r"F:\DeepValve\Bionector\Overlays\Videos\OVERLAYS_0141"
os.makedirs(output_folder, exist_ok=True)  # Crée le dossier de sortie s'il n'existe pas

# Définir les seuils pour détecter la couleur de fond proche de #CAB196
lower_color = np.array([0, 10, 180])  # Minimum en HSV
upper_color = np.array([100, 100, 255])  # Maximum en HSV

# Parcourir toutes les images du dossier
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Filtrer les formats d'image
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Erreur : Impossible de charger l'image {image_name}.")
            continue

        # Convertir en espace HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Créer le masque pour détecter le fond de couleur
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Inverser le masque (zones détectées deviennent blanches)
        mask_inv = cv2.bitwise_not(mask)

        # Conserver uniquement l'objet en utilisant le masque inversé
        result = cv2.bitwise_and(image, image, mask=mask_inv)

        # Ajouter un canal alpha pour la transparence
        b, g, r = cv2.split(result)
        alpha = mask_inv
        result_with_alpha = cv2.merge((b, g, r, alpha))

        # Sauvegarder l'image résultante avec transparence
        output_path = os.path.join(output_folder, os.path.splitext("0141_" + image_name)[0] + ".png")
        cv2.imwrite(output_path, result_with_alpha)
        print(f"Image sauvegardée : {output_path}")

print(f"Traitement terminé. Les images filtrées sont enregistrées dans : {output_folder}")