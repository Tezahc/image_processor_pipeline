import random
from pathlib import Path
from typing import Optional, Dict
from PIL import Image


def process_rotations(
    input_path: Path,
    num_rotations: int = 10,
    include_original: bool = True
) -> Optional[Dict[str, Image.Image]]:
    """
    Charge une image et génère plusieurs rotations aléatoires de celle-ci.

    Args:
        input_path (Path): Chemin vers l'image d'entrée.
        num_rotations (int): Nombre de rotations aléatoires à générer.
        include_original (bool): Si True, inclut également l'image originale
                                 (convertie en RGBA) dans le dictionnaire retourné
                                 sous la clé 'original'.

    Returns:
        Optional[Dict[str, Image.Image]]: Un dictionnaire où les clés sont
        'r001', 'r002', ..., 'rN' (et potentiellement 'original' = r000) et les valeurs
        sont les objets PIL.Image correspondants (en RGBA).
        Retourne None si l'image ne peut pas être ouverte.
    """
    try:
        # Charger l'image et s'assurer qu'elle est en RGBA pour la rotation
        # La conversion est faite ici pour être cohérente entre l'original et les rotations
        img = Image.open(input_path).convert("RGBA")
        width, height = img.size # Obtenir les dimensions originales

        results: Dict[str, Image.Image] = {}

        # Ajouter l'original si demandé
        if include_original:
            results['r000'] = img.copy() # Copie pour éviter modif accidentelle

        # Générer les rotations
        for i in range(num_rotations):
            # Générer un angle aléatoire entre 1 et 359 degrés (0 est l'original)
            # On pourrait aussi faire 0-360, mais 0 est redondant si 'original' est inclus.
            angle = random.uniform(1, 359)

            # Effectuer la rotation
            # expand=True : agrandit la taille de l'image pour contenir toute l'image tournée
            # fillcolor=(0,0,0,0) : remplit le fond ajouté avec du transparent (pour RGBA)
            rotated = img.rotate(angle, expand=True, fillcolor=None) # resample=Image.Resampling.BICUBIC est optionnel pour qualité

            # Recadrer pour enlever le maximum de fond transparent ajouté
            # getbbox() trouve la boîte englobante du contenu non-transparent (alpha > 0)
            bbox = rotated.getbbox()

            # Si bbox est None, cela peut arriver si l'image est complètement transparente
            # ou si une erreur se produit. On garde l'image tournée telle quelle.
            if bbox:
                cropped = rotated.crop(bbox)
                # Vérifier si le recadrage n'a pas résulté en une image de taille nulle
                if cropped.width > 0 and cropped.height > 0:
                     results[f'r{i+1:03d}'] = cropped
                else:
                     print(f"Avertissement [Rotation {input_path.name}]: Recadrage après rotation {i+1} a produit une image vide. Utilisation de l'image tournée non recadrée.")
                     results[f'r{i+1:03d}'] = rotated # Sauvegarde non recadrée si le crop échoue
            else:
                 print(f"Avertissement [Rotation {input_path.name}]: Impossible d'obtenir BBox après rotation {i+1}. Utilisation de l'image tournée non recadrée.")
                 results[f'r{i+1:03d}'] = rotated # Sauvegarde non recadrée

        return results

    except FileNotFoundError:
        print(f"Erreur [Rotation]: Fichier non trouvé {input_path}")
        return None
    except Exception as e:
        print(f"Erreur [Rotation] lors du traitement de {input_path.name}: {e}")
        return None
