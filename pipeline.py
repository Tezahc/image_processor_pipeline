from typing import Callable, Optional, Union, List, Dict, Any
from pathlib import Path
import cv2
from .utils import utils as u
import numpy as np
from PIL import Image


class ProcessingStep:
    """Représente une étape de traitement dans le pipeline."""
    def __init__(self,
                 name: str,
                 process_function: Callable,
                 output_dirs: List[str | Path], # Doit avoir au moins un output pour être utile
                 input_dirs: Optional[List[str | Path]] = None,
                 fixed_input: bool = False, # Si True, input_dirs ne sera pas modifié par le chaînage
                 root_dir: Optional[str | Path] = None,
                 options: Optional[Dict] = None):
        """
        Initialise une étape de traitement.

        Args:
            name (str): Nom de l'étape (pour l'affichage).
            process_function (Callable): La fonction qui traite *un* élément (ex: une image).
                                         Doit prendre au moins un Path en premier argument.
            input_dirs (Optional[List[str | Path]]): Liste des chemins des dossiers d'entrée.
                                                            Si None ou [], sera déterminé par chaînage
                                                            dans le pipeline (sauf si fixed_input=True).
            output_dirs (List[str | Path]): Liste des chemins des dossiers de sortie. Requis.
            fixed_input (bool): Si True, les input_dirs définis ici ne seront pas
                                écrasés par le chaînage automatique. Défaut False.
            root_dir (Optional[str | Path]): Dossier racine pour résoudre les chemins relatifs.
            options (Optional[Dict]): Arguments (kwargs) additionnels à passer à process_function.
        """
        self.name = name
        self.process_function = process_function
        self.root_dir = Path(root_dir) if root_dir else None
        self.fixed_input = fixed_input
        self.process_kwargs = options or {}

        # Initialisation des listes de chemins (sera finalisé dans le pipeline)
        self.input_paths: List[Path] = self._resolve_paths(input_dirs or [])
        self.output_paths: List[Path] = self._resolve_paths(output_dirs) # Output est requis

        if not self.output_paths:
             raise ValueError(f"L'étape '{self.name}' doit avoir au moins un 'output_dirs' défini.")

        self.processed_files_map: Dict[Path, List[Path]] = {} # Stocke input_file -> [output_files]

    def _resolve_paths(self, dir_list: List[str | Path]) -> List[Path]:
        """Convertit les strings en Path et les résout par rapport au root_dir si nécessaire."""
        resolved = []
        for d in dir_list:
            path = Path(d)
            if self.root_dir and not path.is_absolute():
                # Vérifie si le chemin relatif existe déjà par rapport au root_dir
                # Ou s'il faut le créer ( mkdir sera fait dans run() )
                resolved.append(self.root_dir / path)
            else:
                resolved.append(path)
        return resolved

    def update_options(self, **new_kwargs):
        """Met à jour les options passées à la fonction de traitement."""
        self.process_kwargs.update(new_kwargs)

    def _save_result(self, result: Any, input_file: Path) -> List[Path]:
        """
        Sauvegarde le résultat de process_function en fonction de son type.
        Gère np.ndarray (via cv2), PIL.Image, et dict (pour sorties multiples).

        Args:
            result: Le résultat retourné par process_function.
            input_file (Path): Le fichier d'entrée original (pour le nommage).

        Returns:
            List[Path]: La liste des chemins des fichiers sauvegardés.
        """
        saved_files = []
        if result is None:
            # La fonction de process a indiqué qu'il n'y a rien à sauvegarder
            return saved_files

        # Utilise le premier dossier de sortie par défaut pour la sauvegarde.
        # Pour des scénarios plus complexes (ex: sauvegarder dans différents output_dirs
        # basé sur le type ou le contenu du résultat), cette logique devrait être étendue.
        if not self.output_paths:
             print(f"Avertissement [{self.name}]: Aucun dossier de sortie défini pour sauvegarder le résultat de {input_file.name}.")
             return saved_files
        output_dir = self.output_paths[0]

        base_name = input_file.stem # Nom du fichier sans extension
        ext = input_file.suffix    # Extension (ex: .png)

        try:
            if isinstance(result, np.ndarray):
                # Cas: résultat unique type OpenCV/Numpy
                output_filename = input_file.name # Garde le nom original par défaut
                output_path = output_dir / output_filename
                success = cv2.imwrite(str(output_path), result)
                if success:
                    saved_files.append(output_path)
                else:
                    print(f"Erreur [{self.name}]: Échec de la sauvegarde OpenCV pour {output_path}")

            elif isinstance(result, Image.Image):
                # Cas: résultat unique type PIL
                output_filename = input_file.name # Garde le nom original par défaut
                output_path = output_dir / output_filename
                # PIL détermine le format depuis l'extension, gère le '.jpg'/'.jpeg'
                save_ext = ext.lower()
                if save_ext == ".jpg": save_ext = ".jpeg" # PIL préfère .jpeg
                # Assure que le format est supporté (on pourrait ajouter plus de checks)
                if save_ext in [".png", ".jpeg", ".bmp", ".tiff"]:
                   result.save(output_path)
                   saved_files.append(output_path)
                else:
                    # Essayer de sauvegarder en PNG par défaut si extension inconnue/non supportée?
                    output_path_png = output_dir / f"{base_name}.png"
                    print(f"Avertissement [{self.name}]: Extension '{ext}' non gérée directement pour PIL. Tentative de sauvegarde en PNG: {output_path_png}")
                    result.save(output_path_png)
                    saved_files.append(output_path_png)


            elif isinstance(result, dict):
                # Cas: Dictionnaire de résultats (comme pour la symétrie)
                # La clé est utilisée comme suffixe/identifiant
                for key, img_data in result.items():
                    # Vérifier le type de chaque élément dans le dict
                    if isinstance(img_data, (np.ndarray, Image.Image)):
                        output_filename = f"{base_name}_{key}{ext}"
                        output_path = output_dir / output_filename
                        
                        # Appel récursif ou duplication de la logique de sauvegarde ?
                        # Ici, on duplique pour la clarté, mais une fonction helper serait mieux
                        if isinstance(img_data, np.ndarray):
                            success = cv2.imwrite(str(output_path), img_data)
                            if success:
                                saved_files.append(output_path)
                            else:
                                print(f"Erreur [{self.name}]: Échec de la sauvegarde OpenCV pour {output_path}")
                        elif isinstance(img_data, Image.Image):
                             try:
                                img_data.save(output_path)
                                saved_files.append(output_path)
                             except Exception as e_pil_dict:
                                print(f"Erreur [{self.name}]: Échec de la sauvegarde PIL pour {output_path} dans le dict: {e_pil_dict}")

                    else:
                        print(f"Avertissement [{self.name}]: Type non géré '{type(img_data)}' trouvé dans le dictionnaire retourné pour l'entrée {input_file.name} (clé: {key}).")

            # TODO: Ajouter d'autres types si nécessaire (ex: listes, tuples)
            # elif isinstance(result, list):
            #     # Comment nommer les fichiers ? index ?
            #     pass

            else:
                # Type de retour non reconnu
                print(f"Avertissement [{self.name}]: Type de retour non géré '{type(result)}' par la fonction de sauvegarde pour l'entrée {input_file.name}.")

        except Exception as e:
            print(f"Erreur [{self.name}]: Exception lors de la tentative de sauvegarde du résultat de {input_file.name}: {e}")

        return saved_files

    def run(self):
        """Exécute l'étape de traitement pour tous les fichiers dans les dossiers d'entrée."""
        self.processed_files_map = {} # Réinitialise le suivi des fichiers traités
        print(f"--- Exécution Étape : {self.name} ---")
        print(f"Entrée(s): {[str(p) for p in self.input_paths]}")
        print(f"Sortie(s): {[str(p) for p in self.output_paths]}")

        # S'assurer que les dossiers de sortie existent
        for output_path in self.output_paths:
            output_path.mkdir(parents=True, exist_ok=True)

        # Logique d'itération simple: traite les fichiers du *premier* dossier d'entrée.
        # Pour gérer plusieurs dossiers d'entrée (ex: fusionner, comparer),
        # cette boucle devrait être adaptée.
        if not self.input_paths:
             print(f"Avertissement [{self.name}]: Aucun dossier d'entrée défini. Étape sautée.")
             return

        input_path = self.input_paths[0]
        if not input_path.is_dir():
            print(f"Erreur [{self.name}]: Le dossier d'entrée spécifié n'existe pas: {input_path}")
            return

        # Lister les fichiers images (ajuster les extensions si besoin)
        image_files = sorted([f for f in input_path.glob('*') if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')])

        if not image_files:
             print(f"Avertissement [{self.name}]: Aucun fichier image trouvé dans {input_path}")
             return

        print(f"Traitement de {len(image_files)} fichier(s) depuis {input_path}...")
        for file in image_files:
            try:
                # Appelle la fonction de traitement atomique
                result = self.process_function(file, **self.process_kwargs)

                # Appelle la méthode interne de sauvegarde
                saved_files = self._save_result(result, file)
                if saved_files:
                     self.processed_files_map[file] = saved_files # Stocke les fichiers générés

            except Exception as e:
                # Erreur durant l'appel à process_function ou _save_result
                # _save_result a déjà son propre logging d'erreur interne
                print(f"Erreur [{self.name}]: Échec du traitement ou de la sauvegarde pour {file.name}: {e}")
                # traceback.print_exc() # Décommenter pour plus de détails de débogage

        print(f"--- Étape {self.name} terminée ---")


class ProcessingPipeline:
    """Orchestre une séquence d'étapes de traitement (ProcessingStep)."""
    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        """
        Initialise le pipeline.

        Args:
            root_dir (Optional[Union[str, Path]]): Dossier racine utilisé pour résoudre
                                                   les chemins relatifs des étapes.
        """
        self.steps: List[ProcessingStep] = []
        self.root_dir = Path(root_dir) if root_dir else Path.cwd() # Utilise CWD si non fourni

    def add_step(self, step: ProcessingStep, position: Optional[int] = None):
        """
        Ajoute une étape au pipeline, gérant le chaînage des entrées/sorties.

        Args:
            step (ProcessingStep): L'étape à ajouter.
            position (Optional[int]): Position à laquelle insérer l'étape (0-based).
                                      Si None, ajoute à la fin.
        """
        # Appliquer le root_dir du pipeline si l'étape n'en a pas ou pour résoudre les chemins
        if not step.root_dir:
            step.root_dir = self.root_dir
        # Re-résoudre les chemins avec le root_dir final (important si root_dir pipeline est différent)
        step.input_paths = step._resolve_paths([str(p) for p in step.input_paths])
        step.output_paths = step._resolve_paths([str(p) for p in step.output_paths])

        # Déterminer la position d'insertion
        insert_at = position if position is not None else len(self.steps)
        if not (0 <= insert_at <= len(self.steps)):
             raise IndexError(f"Position d'insertion invalide : {insert_at}. Doit être entre 0 et {len(self.steps)}.")

        # Logique de chaînage avant l'insertion
        previous_step = self.steps[insert_at - 1] if insert_at > 0 else None
        next_step = self.steps[insert_at] if insert_at < len(self.steps) else None

        # 1. Définir l'input de la nouvelle étape (si nécessaire)
        if not step.input_paths and not step.fixed_input:
            if previous_step:
                if not previous_step.output_paths:
                     raise ValueError(f"Impossible de chaîner l'étape '{step.name}': "
                                      f"l'étape précédente '{previous_step.name}' n'a pas de dossier de sortie.")
                # Prend le premier dossier de sortie de l'étape précédente par défaut
                step.input_paths = [previous_step.output_paths[0]]
                print(f"Info [{step.name}]: Dossier d'entrée automatiquement défini à '{step.input_paths[0]}' "
                      f"depuis l'étape précédente '{previous_step.name}'.")
            elif insert_at == 0:
                 # C'est la première étape, elle doit avoir un input défini explicitement
                 raise ValueError(f"L'étape '{step.name}' est la première étape ou insérée en position 0, "
                                  "ses 'input_dirs' doivent être définis explicitement.")

        # Insérer l'étape
        self.steps.insert(insert_at, step)

        # 2. Mettre à jour l'input de l'étape suivante (si elle existe et n'est pas fixe)
        if next_step and not next_step.fixed_input:
            # L'étape qu'on vient d'insérer devient la "précédente" pour next_step
            if not step.output_paths:
                 raise ValueError(f"Impossible de mettre à jour l'étape suivante '{next_step.name}': "
                                  f"l'étape insérée '{step.name}' n'a pas de dossier de sortie.")
            next_step.input_paths = [step.output_paths[0]] # Met à jour avec la sortie de la nouvelle étape
            print(f"Info [{next_step.name}]: Dossier d'entrée mis à jour à '{next_step.input_paths[0]}' "
                  f"suite à l'insertion de l'étape '{step.name}'.")

        # Vérification finale après ajout/insertion
        if not step.input_paths and insert_at == 0:
             raise ValueError(f"Erreur fatale: La première étape '{step.name}' n'a toujours pas de dossier d'entrée défini.")


    def run(self, from_step_index: int = 0, only_one_step: bool = False):
        """
        Exécute le pipeline à partir d'une étape donnée.

        Args:
            from_step_index (int): Index de l'étape de départ (0-based).
            only_one_step (bool): Si True, exécute seulement l'étape `from_step_index`.
        """
        if not self.steps:
            print("Le pipeline est vide, rien à exécuter.")
            return

        if not (0 <= from_step_index < len(self.steps)):
            raise IndexError(f"Index de départ invalide {from_step_index}. "
                             f"Le pipeline a {len(self.steps)} étapes (indices 0 à {len(self.steps)-1}).")

        # Détermine les étapes à exécuter
        start_index = from_step_index
        end_index = start_index + 1 if only_one_step else len(self.steps)

        print(f"\n=== Démarrage de l'exécution du Pipeline ===")
        if only_one_step:
            print(f"Exécution de l'étape {start_index} uniquement.")
        else:
            print(f"Exécution des étapes de {start_index} à {end_index - 1}.")

        for i in range(start_index, end_index):
            step = self.steps[i]
            # Vérification simple: le dossier d'entrée existe-t-il ?
            # Utile surtout quand on ne commence pas à l'étape 0.
            if step.input_paths: # Vérifier seulement s'il y a des inputs définis
                input_path_to_check = step.input_paths[0] # Vérifie le premier input par défaut
                if not input_path_to_check.exists() or not input_path_to_check.is_dir():
                     print(f"Avertissement: Le dossier d'entrée principal '{input_path_to_check}' "
                           f"pour l'étape {i} ('{step.name}') n'existe pas. "
                           "Assurez-vous que les étapes précédentes ont été exécutées correctement.")
                     # Optionnel: lever une exception ici si on veut être strict
                     # raise FileNotFoundError(f"Dossier d'entrée requis '{input_path_to_check}' non trouvé pour l'étape {i} ('{step.name}')")
            else:
                 # Normalement impossible si add_step fait bien son travail, sauf pour des étapes "source" spéciales
                 print(f"Avertissement: L'étape {i} ('{step.name}') n'a pas de dossier d'entrée configuré.")


            step.run() # L'étape exécute son traitement

        print(f"=== Exécution du Pipeline terminée ===")
