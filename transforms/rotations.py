import os
import random
from PIL import Image
from tqdm import tqdm
import time
import concurrent.futures

# Dossiers
RACINE = r"C:\Users\GuillaumeChazet\Documents\ICUREsearch\DeepValve\Bionector\Overlays\Photos"
source_dir = os.path.join(RACINE, "symétries")
target_dir = os.path.join(RACINE, "rotations")

# Créer le dossier cible s'il n'existe pas
os.makedirs(target_dir, exist_ok=True)

def process_single_image(source_path, target_dir, num_rotations=100):
    """
    Traite une seule image : la copie, puis génère et sauvegarde ses rotations.
    Conçue pour être exécutée dans un processus séparé.
    """
    try:
        file_name = os.path.basename(source_path)
        target_path = os.path.join(target_dir, file_name)

        # Charger l'image une seule fois
        img = Image.open(source_path).convert("RGBA") # Convertir en RGBA

        # 1. Copier l'image d'origine dans le dossier cible
        #    Note: Si vous ne voulez PAS copier l'original, commentez la ligne suivante.
        img.save(target_path, "png")

        # 2. Effectuer les rotations
        count = 0
        for i in range(num_rotations):
            # Générer un angle aléatoire entre 0 et 360 degrés
            angle = random.uniform(0, 360)

            # Effectuer la rotation (expand=True pour éviter de couper l'image)
            rotated = img.rotate(angle, expand=True) #, resample=Image.Resampling.BICUBIC) # BICUBIC pour meilleure qualité

            # Recadrer l'image pour supprimer les zones vides (transparentes) ajoutées par expand=True
            # getbbox() trouve la boîte englobante du contenu non transparent
            bbox = rotated.getbbox()
            cropped = rotated.crop(bbox)

            # Enregistrer l'image avec le préfixe
            rotated_name = os.path.join(target_dir, f"r{i+1}_{file_name}")
            cropped.save(rotated_name, "png")
            count += 1

        # Retourner le nom du fichier et le nombre de rotations réussies
        return file_name, count
    
    except Exception as e:
        # En cas d'erreur (ex: fichier corrompu), logguer et continuer
        print(f"Erreur lors du traitement de {os.path.basename(source_path)}: {e}")
        return os.path.basename(source_path), 0 # Retourne 0 rotation pour ce fichier

if __name__ == '__main__':
    MAX_WORKERS = 8 # os.cpu_count() # -> 16
    NUM_ROTATIONS = 160

    print(f"Dossier source : {source_dir}")
    print(f"Dossier cible  : {target_dir}")
    print(f"Nombre de rotations par image : {NUM_ROTATIONS}")
    print(f"Utilisation de {MAX_WORKERS} processus parallèles.")

    # Lister les fichiers PNG à traiter
    files_to_process = [
        os.path.join(source_dir, file_name)
        for file_name in os.listdir(source_dir)
        if file_name.lower().endswith(".png")
    ]

    if not files_to_process:
        print("Aucun fichier PNG trouvé dans le dossier source.")

    print(f"{len(files_to_process)} images PNG à traiter...")

    start_time = time.time()
    processed_count = 0

    # Utiliser ProcessPoolExecutor pour la parallélisation
    # 'with' s'assure que le pool est correctement fermé à la fin
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Soumettre toutes les tâches au pool
        # future_to_path = {executor.submit(process_single_image, path, target_dir, NUM_ROTATIONS): path for path in files_to_process}
        futures = [executor.submit(process_single_image, path, target_dir, NUM_ROTATIONS) for path in files_to_process]

        # Utiliser tqdm pour afficher la progression au fur et à mesure que les tâches se terminent
        for future in concurrent.futures.as_completed(futures):
            # print(future)
            try:
                # Récupérer le résultat de la tâche terminée
                file_name, rotations_count = future.result()
                if rotations_count > 0:
                    processed_count += 1
                # Optionnel: Afficher un message pour chaque fichier terminé (peut ralentir si beaucoup de fichiers)
                # print(f"Terminé: {file_name} ({rotations_count} rotations)")
            except Exception as exc:
                # Gérer les exceptions qui pourraient survenir pendant l'exécution de la tâche
                # Bien que process_single_image ait son propre try/except, une erreur de 'pickle' ou autre pourrait survenir ici.
                print(f'Une tâche a généré une exception: {exc}')


    end_time = time.time()
    duration = end_time - start_time
    total_rotations_saved = len(os.listdir(target_dir))

    print("-" * 30)
    print(f"Traitement terminé en {duration:.2f} secondes.")
    print(f"{processed_count} fichiers images traités.")
    print(f"{total_rotations_saved} rotations sauvegardées dans {target_dir}")
    print("-" * 30)