import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Union # Ajout des types nécessaires
from PIL import Image, UnidentifiedImageError # Garder PIL

def process_rotations(
    input_path: Path,
    output_dirs: List[Path],
    # Options spécifiques passées via **options
    num_rotations: int = 10,
    include_original: bool = True,
    angle_min: float = 1.0, # Angle minimum (inclus)
    angle_max: float = 359.0, # Angle maximum (inclus)
    output_format: str = "png", # Format de sortie (ex: PNG, JPEG)
    output_prefix: str = "r", # Préfixe pour les rotations
    original_key: str = "r000", # Clé/Préfixe pour l'original si inclus
    rotation_key_format: str = "{prefix}{index:03d}", # Format pour clé/préfixe rotation
    **options: Any # Accepter d'autres options non utilisées
) -> Optional[List[Path]]: # Retourne List[Path] ou None
    """
    Charge une image, génère plusieurs rotations aléatoires et les SAUVEGARDE.

    Utilise PIL pour charger, tourner (avec expansion et remplissage transparent)
    et recadrer les images. Sauvegarde les images résultantes dans le premier
    dossier de sortie fourni (`output_paths[0]`).

    Args:
        input_path (Path): Chemin vers l'image d'entrée.
        output_paths (List[Path]): Liste des chemins des dossiers de sortie. Le premier est utilisé.
        num_rotations (int): Nombre de rotations aléatoires à générer.
        include_original (bool): Si True, sauvegarde aussi l'image originale (convertie RGBA).
        angle_min (float): Angle de rotation aléatoire minimum (inclus).
        angle_max (float): Angle de rotation aléatoire maximum (inclus).
        output_format (str): Format de sauvegarde PIL (ex: "PNG", "JPEG").
        output_prefix (str): Préfixe utilisé avant l'index de rotation dans le nom de fichier.
        original_key (str): Préfixe/Clé utilisé pour l'image originale si sauvegardée.
        rotation_key_format (str): Format string pour générer le préfixe de rotation
                                   (doit inclure {prefix} et {index}).
        **options (Any): Accepte d'autres options (non utilisées ici).

    Returns:
        Optional[List[Path]]:
            - List[Path]: Liste des chemins complets vers les fichiers sauvegardés.
            - None: Si erreur (lecture, dossier sortie manquant, aucune sauvegarde réussie).
    """
    # --- 1. Vérifications Préliminaires ---
    if not output_dirs:
        print(f"Erreur [{input_path.name} - Rotation]: Aucun dossier de sortie ('output_paths') fourni.")
        return None
    target_dir = output_dirs[0]

    # --- 2. Lecture et préparation de l'image ---
    try:
        # Charger et convertir en RGBA pour gérer la transparence pendant la rotation
        img = Image.open(input_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Erreur [{input_path.name} - Rotation]: Fichier non trouvé.")
        return None
    except UnidentifiedImageError:
         print(f"Erreur [{input_path.name} - Rotation]: Impossible d'identifier ou d'ouvrir l'image (format invalide?).")
         return None
    except Exception as e:
        print(f"Erreur [{input_path.name} - Rotation]: Échec lors de la lecture du fichier: {e}")
        return None

    # --- 3. Génération et Sauvegarde ---
    saved_files: List[Path] = []
    base_name = input_path.stem
    # Déterminer l'extension de sortie en fonction du format demandé
    # (PIL gère la conversion lors de la sauvegarde)
    out_suffix = f".{output_format.lower()}"
    if output_format.lower() == "jpeg": out_suffix = ".jpg" # Convention commune

    # print(f"Info [{input_path.name} - Rotation]: Génération et sauvegarde de {num_rotations} rotations (+original={include_original}) dans {target_dir}...")

    # Sauvegarder l'original si demandé
    if include_original:
        output_filename_orig = f"{original_key}_{base_name}{out_suffix}"
        output_file_path_orig = target_dir / output_filename_orig
        try:
            img.save(output_file_path_orig, format=output_format)
            saved_files.append(output_file_path_orig)
        except Exception as e_save:
            print(f"Erreur [{input_path.name} - Rotation]: Échec sauvegarde de l'original '{output_filename_orig}': {e_save}")
            # On continue même si l'original échoue

    # Générer et sauvegarder les rotations
    for i in range(num_rotations):
        angle = random.uniform(angle_min, angle_max)
        rotated_image: Optional[Image.Image] = None # Pour stocker l'image à sauvegarder

        try:
            # Rotation avec expansion
            # fillcolor=None utilise la couleur par défaut (noir) ou le type si spécifié plus tard
            # mais pour RGBA, (0,0,0,0) est mieux pour un fond transparent
            rotated = img.rotate(angle, expand=True) # resample=Image.Resampling.BICUBIC) # BICUBIC pour meilleure qualité

            # Recadrage
            bbox = rotated.getbbox()
            if bbox:
                cropped = rotated.crop(bbox)
                if cropped.width > 0 and cropped.height > 0:
                    rotated_image = cropped
                else:
                    print(f"Avertissement [{input_path.name} - Rotation]: Recadrage après rotation {i+1} vide. Utilisation de l'image non recadrée.")
                    rotated_image = rotated
            else:
                 print(f"Avertissement [{input_path.name} - Rotation]: Impossible d'obtenir BBox après rotation {i+1}. Utilisation de l'image non recadrée.")
                 rotated_image = rotated

            # Si on a une image à sauvegarder
            if rotated_image:
                # Formatage de la clé/préfixe (ex: r001, r002...)
                rotation_key = rotation_key_format.format(prefix=output_prefix, index=i+1)
                output_filename_rot = f"{base_name}_{rotation_key}{out_suffix}"
                output_file_path_rot = target_dir / output_filename_rot

                # Sauvegarde
                rotated_image.save(output_file_path_rot, format=output_format)
                saved_files.append(output_file_path_rot)

        except Exception as e_rot_save:
            # Attraper les erreurs pendant la rotation ou la sauvegarde de CETTE itération
            print(f"Erreur [{input_path.name} - Rotation]: Échec lors de la génération/sauvegarde de la rotation {i+1} (angle {angle:.1f}°): {e_rot_save}")
            # On continue avec la rotation suivante

    # --- 4. Retour ---
    if not saved_files:
        print(f"Avertissement [{input_path.name} - Rotation]: Aucune image (originale ou rotation) n'a pu être sauvegardée.")
        return None

    # print(f"Info [{input_path.name} - Rotation]: {len(saved_files)} image(s) sauvegardée(s).")
    return saved_files # Retourne la liste des chemins créés