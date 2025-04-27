import cv2
import numpy as np
import matplotlib.pyplot as plt

# Chemin de l'image à tester
image_path = r"C:\Users\GuillaumeChazet\Documents\ICUREsearch\DeepValve\MicroClave\video\CROPPED_0026\frame_00085.jpg"

# Charger l'image
image = cv2.imread(image_path)
if image is None:
    print("Erreur : Impossible de charger l'image.")
    exit()

# Convertir en espace HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def adjust_and_show(h1, s1, v1, h2, s2, v2):
    """
    Fonction pour ajuster les seuils et afficher le résultat
    """
    # Définir les seuils
    lower_white = np.array([h1, s1, v1])  # Plage minimale
    upper_white = np.array([h2, s2, v2])  # Plage maximale

    # Créer le masque
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Appliquer le masque à l'image originale
    result = cv2.bitwise_and(image, image, mask=mask)

    # Afficher les résultats avec Matplotlib
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Image Originale")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title(f"Masque (H: {h1}, S: {s1}, V: {v1})")
    ax[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Résultat")
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation avec différents seuils
# adjust_and_show(0, 0, 0, 180, 50, 100)  # Noir
# adjust_and_show(18, 135, 180, 35, 255, 255)  # Jaune
adjust_and_show(25, 25, 160, 65, 135, 255)  # Jaune
# adjust_and_show(52, 0, 222, 65, 55, 255) # Vert
