from pathlib import Path
import random
from typing import Optional, List, Any

import albumentations as A
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError # Garder PIL
from ultralytics.data.utils import IMG_FORMATS

from ipp.utils import utils
from ipp.utils.artifact import Artifact
from ipp.utils.yolo_labels import YoloLabelHandler


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
    target_dir = utils._validate_dirs(output_dirs, 1)

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
    artifacts: List[Artifact] = []
    base_name = input_path.stem
    # Déterminer l'extension de sortie en fonction du format demandé
    # (PIL gère la conversion lors de la sauvegarde)
    out_suffix = f".{output_format.lower()}"
    if output_format.lower() == "jpeg": out_suffix = ".jpg" # Convention commune

    # print(f"Info [{input_path.name} - Rotation]: Génération et sauvegarde de {num_rotations} rotations (+original={include_original}) dans {target_dir}...")

    # Sauvegarder l'original si demandé
    if include_original:
        output_filename_orig = f"{base_name}_{original_key}{out_suffix}"
        output_file_path_orig = target_dir / output_filename_orig
        try:
            img.save(output_file_path_orig, format=output_format)
            output_orig = Artifact(output_file_path_orig, transformation="rotation", params={"angle": 0, "name_id": 0})
            artifacts.append(output_file_path_orig)
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
                artifacts.append(output_file_path_rot)

        except Exception as e_rot_save:
            # Attraper les erreurs pendant la rotation ou la sauvegarde de CETTE itération
            print(f"Erreur [{input_path.name} - Rotation]: Échec lors de la génération/sauvegarde de la rotation {i+1} (angle {angle:.1f}°): {e_rot_save}")
            # On continue avec la rotation suivante

    # --- 4. Retour ---
    if not artifacts:
        print(f"Avertissement [{input_path.name} - Rotation]: Aucune image (originale ou rotation) n'a pu être sauvegardée.")
        return None

    # print(f"Info [{input_path.name} - Rotation]: {len(saved_files)} image(s) sauvegardée(s).")
    return artifacts # Retourne la liste des chemins créés


def rotate_image_with_labels(
    *input_paths: Path,
    output_dirs: List[Path],
    num_rotations: int = 10,
    include_original: bool = True,
    angle_min: float = -30,
    angle_max: float = 30,
    crop_border: bool = False,
    seed: Optional[int] = None,
    **options: Any
) -> Optional[list[Artifact]]:
    """
    Génère des rotations aléatoires d'une image, avec prise en charge optionnelle
    des labels YOLO via Albumentations.

    - Les bounding boxes restent axis-aligned.
    - Chaque sortie est décrite par un Artifact.
    - Compatible avec ou sans labels.

    Parameters
    ----------
    *input_paths : Path
        input_paths[0] = image
        input_paths[1] (optionnel) = label YOLO
    output_dirs : list[Path]
        output_dirs[0] = images
        output_dirs[1] (optionnel) = labels
    num_rotations : int
        Nombre de rotations aléatoires.
    include_original : bool
        Inclure l'image originale sans transformation.
    angle_min, angle_max : float
        Bornes des angles de rotation (en degrés).
    seed : Optional[int]
        Seed aléatoire pour reproductibilité.

    Returns
    -------
    Optional[List[Artifact]]
    """
    image_path = input_paths[0]
    label_path = input_paths[1] if len(input_paths) > 1 else None

    if image_path.suffix.lower()[1:] not in IMG_FORMATS:
        raise ValueError(f"Format non supporté : {image_path.name}")

    has_labels = label_path is not None

    #TODO: voir avec utils._validate_dirs ? et nb_dirs conditionnel ?
    if not output_dirs:
        raise ValueError(f"Erreur [{image_path.name}] : aucun dossier de sortie ('output_dirs') fourni.")
    if has_labels and len(output_dirs) < 2:
        raise ValueError(
            f"Erreur [{image_path.name}] : labels fournis mais un seul dossier "
            "de sortie configuré. Deux requis (images, labels)."
        )
    image_out_dir = output_dirs[0]
    label_out_dir = output_dirs[1] if has_labels else None

    # choix d'une seed pour permettre une reconstruction
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Crée la liste des angles à appliquer et gère l'ajout de l'original
    rng = random.Random(seed)
    angles = [rng.uniform(angle_min, angle_max) for _ in range(num_rotations)]
    if include_original:
        angles.append(0.0)
    
    trace = {
        "num_rotations": num_rotations,
        "angle_bounds": (angle_min, angle_max),
        "crop_border": crop_border,
        "seed": seed,
        "applied_angles": angles
    }

    # --- Chargement de l'image ---
    image = utils._load_image(image_path)
    # Albumentations travaille en RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # --- Chargement des labels ---
    handler = None
    compose_kwargs: dict = {}
    call_kwargs_labels: dict = {}
    meta: dict = {}

    if has_labels:
        handler = YoloLabelHandler.from_file(label_path, img_h, img_w)
        compose_kwargs, call_kwargs_labels, meta = handler.to_albumentations(
            img_w, img_h,
            use_masks=True  # rotation libre : les polygones peuvent être découpés par le bord,
                            # un masque → raster → contour donne un résultat géométriquement correct.
                            # Les keypoints clippés distordent le polygone.
        )
    
    artifacts: List[Artifact] = []

    # --- Albumentations transform ---
    rotate_tf = A.Rotate(limit=(angle_min, angle_max),
                         border_mode=cv2.BORDER_REPLICATE,
                         rotate_method="ellipse",
                         crop_border=crop_border,
                         p=1.0)
    bbox_params = A.BboxParams(coord_format="yolo",
                               label_fields=["class_labels"],
                               min_visibility=0.0)
    transform = A.Compose([rotate_tf], **compose_kwargs)

    
    # --- Rotations ---
    for idx, angle in enumerate(angles):
        # is_original = angle == 0.0
        is_original = (angle == 0.0) #include_original
        
        if is_original:
            rotated_image = image
            transformed = {"image": image, **call_kwargs_labels}
        else:
            rotated = transform(image=image, **call_kwargs_labels)
            rotated_image = rotated["image"]
            transformed = rotated

        # setdefault permet de prendre cette valeur si l'arg n'est pas fourni. 
        # Mais on peut toujours l'écraser en le précisant.
        # /!\ suffix_index, s'il est précisé, est passé via les kwargs (**options)
        options.setdefault("suffix_key", "r")
        out_img_path = utils.build_output_filepath(image_path, image_out_dir, idx=idx, **options)

        cv2.imwrite(str(out_img_path), cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR))

        artifact = Artifact(out_img_path,
                            transformation="rotation",
                            params={"angle":angle, "seed":seed, "index": idx})
        
        if handler is not None and label_out_dir:
            out_lbl_path = utils.build_output_filepath(label_path, label_out_dir, idx=idx, **options)
            img_h_t, img_w_t = rotated_image.shape[:2]
            out_handler = YoloLabelHandler(img_h_t, img_w_t)

            # Bbox
            out_handler.update_bboxes(
                transformed.get("bboxes", []),
                transformed.get("bboxes_classes", []),
                replace=False
            )

            # Segmentation (reconstruite depuis les masques transformés)
            out_handler.update_from_masks(
                transformed.get("masks", []),
                meta.get("seg_classes", []),
                replace=False
            )
            out_handler.save(out_lbl_path)
            artifact.label_path = out_lbl_path

        artifacts.append(artifact)
    
    return artifacts if artifacts else None