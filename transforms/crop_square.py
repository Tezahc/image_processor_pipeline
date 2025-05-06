import cv2
import random
import numpy as np
from warnings import warn
from pathlib import Path
from typing import Any, List, Optional, Tuple
# import albumentations as A
from albumentations.augmentations.crops.functional import crop as albumentations_crop
from albumentations.core.bbox_utils import (
    convert_bboxes_to_albumentations,
    convert_bboxes_from_albumentations,
    denormalize_bboxes,
    normalize_bboxes,
    # check_bbox, # Utile pour valider la bbox avant conversion
)
from icecream import ic

def convert_bbox(bboxes, shape: Tuple[int, int], to_format: str = "albus"):
        """
        Convertit une ou plusieurs bbox entre les formats YOLO et Albumentations (absolut).
        
        Args:
            bbox (list or np.ndarray): La bounding box à convertir.
            to_format (str): Format cible, "albus" ou "yolo".

        Returns:
            np.ndarray: Bounding box convertie.
        """
        if str.lower(to_format) == 'albus':
            bboxes_absolute = denormalize_bboxes(
                convert_bboxes_to_albumentations(
                    np.array(bboxes), 
                    source_format = 'yolo',
                    shape = shape
                ),
                shape = shape
            )
            return bboxes_absolute

        elif str.lower(to_format) == 'yolo':
            bbox_yolo = convert_bboxes_from_albumentations(
                normalize_bboxes(
                    bboxes=np.array(bboxes), 
                    shape=shape
                ),
                target_format = 'yolo',
                shape = shape
            )
            return bbox_yolo
        
        else:
            raise AttributeError("Format non pris en charge pour la conversion, Veuillez choisir entre ['albus', 'yolo'] (insenssible à la casse)")


def process_square_crop_around_bbox(
    input_image_path: Path,
    input_label_path: Path,
    output_dirs: List[Path],
    # options utiles
    output_prefix: str = "crop_", # Préfixe pour les noms de fichiers
    output_format_image: str = "JPEG", # Format pour l'image cropée
    **options: Any # Accepter d'autres options non utilisées
) -> Optional[List[Path]]: # Retourne une liste de 2 Path (image, label) ou None
    """
    Effectue un crop carré aléatoire autour de la BBox d'une image et sauvegarde
    l'image recadrée ainsi que le fichier label YOLO mis à jour.

    La position du crop carré (dont la taille est min(largeur, hauteur) de l'image)
    est choisie aléatoirement de sorte que la bounding box originale soit
    entièrement contenue dans la zone recadrée.

    Args:
        input_image_path (Path): Chemin vers l'image d'entrée.
        output_paths (List[Path]): Liste des chemins des dossiers de sortie.
                                   Attend au moins 2: [0] pour images, [1] pour labels.
        label_dir_name (str): Nom du dossier (relatif au parent de l'image) où trouver
                              le fichier label d'entrée et où sauvegarder le label de sortie.
                              Ex: si image est 'data/images/img.jpg', label sera cherché/créé
                              dans 'data/labels/img.txt'.
        output_prefix (str): Préfixe pour les noms des fichiers de sortie.
        output_format_image (str): Format de sauvegarde PIL/OpenCV pour l'image (ex: JPEG, PNG).
        **options (Any): Accepte d'autres options (non utilisées ici).


    Returns:
        Optional[List[Path]]:
            - List[Path]: Liste contenant [chemin_image_sauvegardée, chemin_label_sauvegardé].
            - None: Si erreur (lecture, label manquant/invalide, dossiers sortie insuffisants,
                    placement impossible, sauvegarde échouée).

    Raises:
        (Peut lever des exceptions internes de OpenCV, PIL ou Albumentations si non gérées)
    """
    # --- 1. Vérifications Préliminaires et Chemins ---
    if len(output_dirs) < 2:
        raise IndexError(f"Erreur [{input_image_path.name} - CropCarre]: "
              f"Au moins 2 dossiers de sortie sont requis (images, labels), {len(output_dirs)} fourni(s).")
    image_target_dir = Path(output_dirs[0])  # images
    label_target_dir = Path(output_dirs[1])  # labels

    if input_image_path.stem != input_label_path.stem:
        warn(f"Warning [Crop Carré]: image ({input_image_path.name}) et label ({input_label_path.name}) n'ont pas le même nom. Poursuite du traitement...")
    
    # --- 2. Lecture Image et Label ---
    try:
        image = cv2.imread(str(input_image_path))
        if image is None:
            raise IOError(f"Erreur [{input_image_path.name} - CropCarre]: Impossible de charger l'image via OpenCV.")
        
        height, width = image.shape[:2]

        if not input_label_path.is_file():
            raise FileNotFoundError(f"Erreur [{input_image_path.name} - CropCarre]: Fichier label attendu non trouvé: {input_label_path}")
        
        bboxes = []
        class_ids = []
        with open(input_label_path, "r") as b:
            for line in b:
                data = line.strip().split()
                class_id = int(data[0])
                cx, cy, bw, bh = map(float, data[1:])

                bboxes.append((cx, cy, bw, bh))
                class_ids.append(class_id)
                if len(data) != 5:
                    raise ValueError(f"Erreur : [{input_image_path.name} - Crop Carré] :"
                                     f"Format label invalide, 5 valeurs attendues par ligne. Reçu {line}")
                if not all(0 <= float(val) <= 1 for val in data[1:]):
                    raise ValueError(f"Erreur [{input_image_path.name} - Crop Carré] : "
                                     f"Coordonnées YOLO invalides (hors [0,1]) dans {input_label_path.name}. Recu {data[1:]}")

    except FileNotFoundError as fnf: # Redondant avec check is_file mais sécurité
        raise FileNotFoundError(f"Erreur [{input_image_path.name} - CropCarre]: Fichier image ou label non trouvé.") from fnf    
    except ValueError as ve: # Erreur map(float,...) ou int(...)
        raise ValueError(f"Erreur [{input_image_path.name} - CropCarre]: Erreur de format dans le fichier label {input_label_path}") from ve
    except Exception as e:
        raise NotImplementedError(f"Erreur [{input_image_path.name} - CropCarre]: Échec du recadrage") from e

    # --- 3. Logique du Crop Carré Aléatoire autour de la BBox ---
    try:
        bboxes_absolute = convert_bbox(np.array(bboxes), image.shape[:2], 'albus')
        # Taille du crop carré
        crop_size = min(height, width)

        x_min = int(min(bbox[0] for bbox in bboxes_absolute))
        y_min = int(min(bbox[1] for bbox in bboxes_absolute))
        x_max = int(max(bbox[2] for bbox in bboxes_absolute))
        y_max = int(max(bbox[3] for bbox in bboxes_absolute))

        # Calculer les bornes pour le coin supérieur gauche (x_start, y_start) du crop
        # pour que la bbox soit contenue.
        lower_bound_x = max(0, x_max - crop_size)
        upper_bound_x = min(x_min, width - crop_size)

        lower_bound_y = max(0, y_max - crop_size)
        upper_bound_y = min(y_min, height - crop_size)

        # Vérifier s'il existe une position valide
        if lower_bound_x > upper_bound_x or lower_bound_y > upper_bound_y:
            print(f"Avertissement [{input_image_path.name} - CropCarre]: Impossible de trouver une position de crop carré "
                  f"contenant entièrement la bbox [{x_min},{y_min},{x_max},{y_max}] dans une image {width}x{height} "
                  f"avec crop_size={crop_size}. Crop annulé.")
            # Optionnel: On pourrait cropper au centre ou autre, mais ici on annule.
            return None

        # Choisir une position aléatoire valide
        x_start = random.randint(lower_bound_x, upper_bound_x)
        y_start = random.randint(lower_bound_y, upper_bound_y)

        # Effectuer le crop sur l'image
        cropped_image = albumentations_crop(image, x_start, y_start, x_start + crop_size, y_start + crop_size)
        crop_height, crop_width = cropped_image.shape[:2]
        if crop_height <= 0 or crop_width <= 0:
             print(f"Erreur [{input_image_path.name} - CropCarre]: Le crop a résulté en une image de taille nulle ou négative.")
             return None

        new_bboxes_absolute = []
        new_class_ids = []
        for bbox, class_id in zip(bboxes_absolute, class_ids):
            # Mettre à jour la bbox pour qu'elle soit relative au crop (en pixels absolus)
            new_x_min = max(0, bbox[0] - x_start)
            new_y_min = max(0, bbox[1] - y_start)
            new_x_max = min(crop_size, bbox[2] - x_start)
            new_y_max = min(crop_size, bbox[3] - y_start)

        # Re-convertir la nouvelle bbox absolue en YOLO normalisé par rapport au crop
            if new_x_min < new_x_max and new_y_min < new_y_max:
                new_bboxes_absolute.append((new_x_min, new_y_min, new_x_max, new_y_max))
                new_class_ids.append(class_id)
        
        new_bboxes = convert_bbox(new_bboxes_absolute, cropped_image.shape[:2], to_format='yolo') # Vérifie si xmin<xmax etc.
        
    except Exception as e_process:
        print(f"Erreur [{input_image_path.name} - CropCarre]: Échec pendant le processus de crop/bbox: {e_process}")
        # import traceback
        # traceback.print_exc() # Pour debug
        return None


    # --- 4. Sauvegarde Image et Label ---
    saved_paths: List[Path] = []

    img_output_path = image_target_dir / input_image_path.name
    label_output_path = label_target_dir / input_label_path.name

    try:
        # Sauvegarder l'image cropée (avec cv2 comme l'original)
        success_img = cv2.imwrite(str(img_output_path), cropped_image)
        if not success_img:
             # Rare, mais peut arriver si imwrite échoue silencieusement
             raise IOError(f"cv2.imwrite a retourné False pour {img_output_path}")
        saved_paths.append(img_output_path)

        # Sauvegarder le nouveau label
        with open(label_output_path, 'w', encoding='utf-8') as b:
            for class_id, bbox in zip(new_class_ids, new_bboxes):
                cx, cy, w, h = tuple(bbox)
                b.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        saved_paths.append(label_output_path)

        # print(f"Info [{input_image_path.name} - CropCarre]: Image cropée et label sauvegardés.")
        # --- 5. Retourner la liste des chemins sauvegardés ---
        return saved_paths

    except Exception as e_save:
        print(f"Erreur [{input_image_path.name} - CropCarre]: Échec lors de la sauvegarde: {e_save}")
        # Nettoyage optionnel des fichiers partiels
        for p in saved_paths:
            try:
                if p.exists(): p.unlink()
            except OSError: pass # Ignorer si le nettoyage échoue
        return None

if __name__ == '__main__':
    process_square_crop_around_bbox(
        Path('crop_carre_test/imgs/AUTOFLUSH32.jpg'),
        Path('crop_carre_test/labels/CARESITE38.txt'),
        ["CropCarre/imgs", "CropCarre/labels"]
    )