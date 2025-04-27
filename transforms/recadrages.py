import cv2
import os

# Chemin des photos
input_folder = r"F:\DeepValve\Bionector\Overlays\Videos\IMAGES_0150"
output_folder = r"F:\DeepValve\Bionector\Overlays\Videos\CROPPED_0150"
os.makedirs(output_folder, exist_ok=True)  # Crée le dossier de sortie s'il n'existe pas

# Pourcentages à recadrer (en % des dimensions de l'image)
crop_top_percent = 30    # Supprime 10% en haut
crop_bottom_percent = 15  # Supprime 10% en bas
crop_left_percent = 5    # Supprime 10% à gauche
crop_right_percent = 5   # Supprime 10% à droite

# Parcourir toutes les images du dossier
for image_name in os.listdir(input_folder):
    if image_name.endswith('.jpg'):  # Filtrer les fichiers .jpg
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Erreur : Impossible de charger l'image {image_name}.")
            continue

        # Dimensions de l'image
        height, width = image.shape[:2]

        # Calcul des pixels à recadrer
        crop_top = int(height * crop_top_percent / 100)
        crop_bottom = int(height * crop_bottom_percent / 100)
        crop_left = int(width * crop_left_percent / 100)
        crop_right = int(width * crop_right_percent / 100)

        # Recadrage de l'image
        cropped_image = image[crop_top:height - crop_bottom, crop_left:width - crop_right]

        # Sauvegarder l'image recadrée
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, cropped_image)
        print(f"Image recadrée et sauvegardée : {output_path}")

print(f"Traitement terminé. Les images recadrées sont enregistrées dans : {output_folder}")