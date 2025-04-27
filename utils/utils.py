from pathlib import Path


def check_path(folder_name, root=None):
    """
    Construit un chemin complet pour un dossier en fonction de son nom et d'un chemin racine optionnel.

    Cette fonction vérifie si le nom de dossier fourni est un chemin absolu ou relatif.
    Si c'est un chemin absolu, il est retourné tel quel. Sinon, il est combiné avec le chemin racine
    spécifié ou le répertoire de travail actuel si aucun chemin racine n'est fourni.

    :param folder_name: Le nom du dossier ou le chemin du dossier à vérifier.
                        Peut être un chemin relatif ou absolu.
    :type folder_name: str ou Path
    :param root: Le chemin racine à utiliser si folder_name est un chemin relatif.
                 Si non spécifié, le répertoire de travail actuel est utilisé.
                 Ignoré si folder_name est un chemin absolu
    :type root: str ou Path, optional
    :return: Le chemin complet du dossier.
    :rtype: Path
    """
    # Convertir folder_name en objet Path
    path = Path(folder_name)

    # Déterminer le chemin racine : utiliser root s'il est fourni, sinon utiliser le répertoire de travail actuel
    root_path = Path(root) if root else Path('.')

    # Vérifier si le chemin est absolu
    if path.is_absolute():
        # Retourner le chemin absolu tel quel
        return path
    else:
        # Combiner le chemin relatif avec le chemin racine
        return root_path / path
