from pathlib import Path
import cv2


def crop_from_border(filename: Path, options: tuple = (0, 0, 0, 0)):
    # Filtrer les fichiers .jpg
    if not filename.suffix == '.jpg':
        raise ValueError("Le Fichier n'est pas du bon type")

    crop_top, crop_bottom, crop_left, crop_right = options

    # lecture de l'image
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Erreur : Impossible de charger l'image {filename}.")

    # Dimensions de l'image
    height, width = image.shape[:2]

    # Calcul des pixels à recadrer
    # TODO: gérer les % ET les valeurs en px
    crop_top = int(height * crop_top) if 0 <= crop_top < 1 else crop_top
    crop_bottom = int(height * crop_bottom) if 0 <= crop_bottom < 1 else crop_bottom
    crop_left = int(width * crop_left) if 0 <= crop_left < 1 else crop_left
    crop_right = int(width * crop_right) if 0 <= crop_right < 1 else crop_right

    # Recadrage de l'image
    cropped_image = image[crop_top:height - crop_bottom, crop_left:width - crop_right]

    return cropped_image
