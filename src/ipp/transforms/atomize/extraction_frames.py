import cv2
import os

# Chemin de la vidéo .MOV
video_path = r'C:\Users\GuillaumeChazet\Documents\ICUREsearch\DeepValve\MicroClave\video\DSC_0026.MOV'

# Dossier de sortie pour les images
output_folder = r'C:\Users\GuillaumeChazet\Documents\ICUREsearch\DeepValve\MicroClave\video\IMAGES_0026'
os.makedirs(output_folder, exist_ok=True)

# Chargement de la vidéo
cap = cv2.VideoCapture(video_path)

# Vérifie si la vidéo est ouverte
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

frame_count = 0
success = True

# Boucle pour lire les images de la vidéo
while success:
    success, frame = cap.read()
    if success:
        frame_name = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_name, frame)
        frame_count += 1

cap.release()
print(f"Extraction terminée. {frame_count} images sauvegardées dans '{output_folder}'.")