import cv2
from pathlib import Path
from typing import Any, List, Optional
from ultralytics.data.utils import VID_FORMATS

def frame_extraction(
    video_path: Path,
    output_dirs: List[Path],
    file_basename: str,
    **options: Any
) -> Optional[List[Path]]:
    """Extraie les frames d'une vidéo"""
    if not file_basename:
        raise ValueError("Aucun nom de fichier de base fournit pour les nom des frames.")

    # Création d'un dossier par fichier, au nom du fichier vidéo d'origine
    output_dir = output_dirs[0] / video_path.stem
    # exceptionnellement on crée un dossier pour chaque fichier input 
    # on le gère donc ici au lieu de l'orchestrateur
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement de la vidéo
    cap = cv2.VideoCapture(str(video_path))
    
    # Vérifie qu'elle est bien ouverte
    if not cap.isOpened():
        raise RuntimeError("Erreur : Impossible d'ouvrir la vidéo")

    # Vérification que c'est un média compatible mais avec les formats ultralytics 
    # or on lui fournit que des images en bout de chaine so...
    if video_path.suffix[1:].lower() not in VID_FORMATS:
        raise ValueError(f"Fichier vidéo {video_path.suffix} non pris en charge." \
                         f"Format autorisés : {VID_FORMATS}")
    
    frame_count = 0
    success = True
    # Boucle sur chaque image de la vidéo
    while success:
        success, frame = cap.read()
        if success:
            # nom composé du basename (nom de classe) suivi du n° de frame
            frame_name = f"{file_basename}-frame_{frame_count:04d}.jpg"
            output_path = output_dir / frame_name
            cv2.imwrite(str(output_path), frame)
            frame_count += 1
    
    cap.release()
    return output_dir