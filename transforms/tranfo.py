import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from typing import Any, List, Optional
from image_processor_pipeline.utils.utils import _validate_dirs, _load_image


def enhance_image(
    input_image: Path,
    apply_blur: bool,
    apply_rgb: bool,
    output_dirs: List[Path],
    **options: Any
) -> Optional[Path]:
    """Applique des transformation d'images sur un sample donné

    Parameters
    ----------
    input_image : Path
        Chemin du fichier image à transformer
    apply_blur : bool
        Booléen renvoyé par le générateur selon le taux de sampling (30%) si le blur deoit être appliqué
    apply_rgb : bool
        idem `apply_blur` pour la transof du filtre RGB
    output_dirs : List[Path]
        Liste des dossiers de destination

    Returns
    -------
    Optional[Path]
        Chemins enregistrés si succès
    """
    destination_img = _validate_dirs(output_dirs, 1)
    output_path = destination_img / input_image.name
    
    with Image.open(input_image).convert("RGB") as img:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))

        if apply_blur:
            blur_radius = random.uniform(0.5, 3)
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))

        if apply_rgb:
            r, g, b = img.split()
            r = r.point(lambda p: max(0, min(255, p * random.uniform(0.75, 1.25))))
            g = g.point(lambda p: max(0, min(255, p * random.uniform(0.75, 1.25))))
            b = b.point(lambda p: max(0, min(255, p * random.uniform(0.75, 1.25))))
            img = Image.merge("RGB", (r, g, b))
        
        img.save(output_path)
    
    return output_path

def gray_world_transform(img:np.ndarray) -> np.ndarray:
    """Applique une balance des blanc selon la méthode "gray world" sur une image cv2.

    Parameters
    ----------
    img : np.ndarray
        Image sous forme de matrice. 3 canaux attendus au format BGR.

    Returns
    -------
    np.ndarray
        Image balancée, au format BGR.
    """
    img_cp = img.copy()
    b_avg, g_avg, r_avg = img_cp.mean(axis=(0,1))
    img_avg = img_cp.mean()

    img_cp[:,:,0] = np.clip(img_cp[:,:,0] * (img_avg / b_avg), 0, 255)
    img_cp[:,:,1] = np.clip(img_cp[:,:,1] * (img_avg / g_avg), 0, 255)
    img_cp[:,:,2] = np.clip(img_cp[:,:,2] * (img_avg / r_avg), 0, 255)
    return img_cp

def preprocess(input_image:Path,
               output_dirs:List[Path],
               **kwargs):
    image_target_dir = _validate_dirs(output_dirs, 1)
    img = _load_image(input_image)

    balanced_image = gray_world_transform(img)
    
    output_path = image_target_dir / input_image.name

    try:
        sucess = cv2.imwrite(str(output_path), balanced_image)
        if sucess:
            return output_path
        else:
            print(f"Avertissement [{input_image.name} - Gray World]: Échec de sauvegarde (imwrite a retourné False) pour {output_path.name}")
            return None
    except Exception as e_save:
        print(f"Erreur [{input_image.name} - Gray World]: Échec de sauvegarde pour {output_path.name}: {e_save}")
        return None
    