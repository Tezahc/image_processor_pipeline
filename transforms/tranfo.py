import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from typing import Any, List, Optional
from image_processor_pipeline.utils.utils import _validate_dirs


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
    destination_img, destination_lbl = _validate_dirs(output_dirs, 2)
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