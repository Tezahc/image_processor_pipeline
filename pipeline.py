from typing import Any, Callable, Optional, List, Union, Tuple, Dict
from pathlib import Path
import cv2
import numpy as np
from .utils import utils as u


class ProcessingStep:
    def __init__(self,
                 name: str,
                 process_function: Callable,
                 output_dir: str,
                 input_dir: str = None,
                 fixed_input: bool = False,
                 root_dir: str = None,
                 options: dict = None):
        self.name = name
        self.process_function = process_function
        self.root_dir = root_dir
        self.input_dir = u.check_path(input_dir, root_dir)
        self.output_dir = u.check_path(output_dir, root_dir)
        self.fixed_input = fixed_input
        self.process_kwargs = options or {}

    def update_options(self, **new_kwargs):
        self.process_kwargs.update(new_kwargs)

    def _save_result(self, result: Any, input_file: Path) -> List[Path]:
        saved_files = []
        if result is None:
            return saved_files
        
        if not self.output_dir:
            print(f"Warning [{self.name}]: Aucun dossier de sortie défini pour sauvegarder le résultat de {input_file.name}.")
            return saved_files
        
        output_dir = self.output_dir

        try:
            if isinstance(result, np.ndarray):
                # Cas : Résultat unique de type OpenCV/Numpy
                output_path = output_dir / input_file.name
                success = cv2.imwrite(str(output_path), result)
                if success:
                    saved_files.append(output_path)
                else:
                    print(f"Erreur [{self.name}]: Échec de la sauvegarde OpenCV pour {output_path.name}")
            
            elif isinstance(result, dict):
                # Cas : Dictionnaire de résultats (symétrie par ex)
                # la clé est utilisée comme suffixe/identifiant
                for key, img_data in result.items():
                    # Vérifier le type de chaque élément dans le dict
                    if isinstance(img_data, np.ndarray):
                        output_filename = f"{input_file.name}_{key}{input_file.suffix}"
                        output_path = output_dir / output_filename

                        success = cv2.imwrite(str(output_path), img_data)
                        if success: 
                            saved_files.append(output_path)
                        else:
                            print(f"Erreur [{self.name}] : Échec de la sauvegarde OpenCV pour {output_path}")
                    else:
                        print(f"Warning [{self.name}] : Type non géré '{type(img_data)}' trouvé dans le dictionnaire retourne pour l'entrée {input_file.name} (clé: {key}).")
        except Exception as e:
            print(f"Erreur [{self.name}] : Exception lors de la tentative de sauvegarde du résultat de {input_file.name} : {e}")
        
        return saved_files

    def run(self, input_dir: Path=None):
        self.processed_files = []
        input_path = Path(input_dir or self.input_dir)
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in input_path.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        for file in image_files:
            try:
            # permet de continuer le traitement en cas d'erreur
                result = self.process_function(file, **self.process_kwargs)
                if result is not None:
                    output_file = output_path / file.name
                    saved_files = self._save_result(result, file)
                    if saved_files:
                        self.processed_files.append(saved_files)

            except Exception as e:
                print(f"Erreur lors du traitement de {file.name}: {e}")


class MultiInputOutputStep(ProcessingStep):
    def run(self):
        input_path1 = Path(self.input_dirs[0])
        input_path2 = Path(self.input_dirs[1])
        output_path1 = Path(self.output_dirs[0])
        output_path2 = Path(self.output_dirs[1])

        output_path1.mkdir(exist_ok=True)
        output_path2.mkdir(exist_ok=True)

        for file1, file2 in zip(input_path1.glob("*"), input_path2.glob("*")):
            res1, res2 = self.process_function(file1, file2)
            res1.save(output_path1 / file1.name)
            res2.save(output_path2 / file2.name)


class ProcessingPipeline:
    def __init__(self, root_dir: Optional[str] = None):
        self.steps: List[ProcessingStep] = []
        # définit le dossier source du pipeline → obligatoire ?
        self.root_dir = Path(root_dir) if root_dir else None

    def add_step(self, step: ProcessingStep, position=None):
        # Vérification : la première étape doit avoir des inputs définis
        if not self.steps and step.input_dir is None:
            raise ValueError(f"The first step ('{step.name}') must have input_dir defined.")
            # self.steps.append(step)
            # return

        # si un dossier racine est défini dans le pipeline, mais pas dans l'étape,
        # il est transmis à l'étape dès son ajout
        if self.root_dir and not step.root_dir:
            step.root_dir = self.root_dir
            # modifie les dossiers d'input/output s'ils sont définis comme des noms de dossier ou des path relatifs

            # TODO: Tester ce check. Peut être pas ici puisqu'on modifie input_dir juste après
            step.input_dir = u.check_path(step.input_dir, self.root_dir)
            step.output_dir = u.check_path(step.output_dir, self.root_dir)

        if position is None or position < 0:
            previous_step = self.steps[-1] if self.steps else None

            if step.input_dir is None:
                if previous_step is None:
                    raise ValueError("The first step must have an input_dir defined.")
                step.input_dir = previous_step.output_dir

            self.steps.append(step)

        else:
            # Insertion à une position donnée
            if position > len(self.steps):
                raise IndexError("Invalid position to insert step.")

            if position == 0:
                raise ValueError("Cannot insert at position 0. Input_dir must be set manually for the first step.")

            previous_step = self.steps[position - 1]
            next_step = self.steps[position] if position < len(self.steps) else None

            # Définir input_dir si non défini à la création de l'étape
            if step.input_dir is None:
                step.input_dir = previous_step.output_dir

            # Insertion
            self.steps.insert(position, step)

            # Mettre à jour input_dir de la prochaine étape si elle n'est pas fixe
            if next_step and not next_step.fixed_input:
                next_step.input_dir = step.output_dir

    def run(self, from_step_index: int = 0, only_one: bool = False):
        if from_step_index < 0 or from_step_index >= len(self.steps):
            raise IndexError(f"Invalid start index {from_step_index}. Pipeline has {len(self.steps)} steps.")
        
        steps_to_do = [self.steps[from_step_index]] if only_one else self.steps[from_step_index:]
        
        for i, step in enumerate(steps_to_do, start=from_step_index):
            print(f"Running étape {i}: {step.name}")
            step.run()


class OverlayAndLabelStep(ProcessingStep):
    """
    Étape de pipeline spécialisée pour superposer des images (overlays)
    sur des fonds (backgrounds) et générer les images résultantes
    ainsi que les fichiers de labels YOLO correspondants.

    Cette étape gère deux dossiers d'entrée et deux dossiers de sortie.
    """
    def __init__(self,
                 name: str,
                 process_function: Callable, # Doit accepter (overlay_path, background_path, **options)
                 overlay_dirs: List[Union[str, Path]],
                 background_dirs: List[Union[str, Path]],
                 image_output_dirs: List[Union[str, Path]],
                 label_output_dirs: List[Union[str, Path]],
                 background_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
                 overlay_extensions: Tuple[str, ...] = ('.png',), # Typiquement PNG pour la transparence
                 output_prefix: str = "BN_", # Préfixe ajouté aux noms de fichiers de sortie
                 root_dir: Optional[Union[str, Path]] = None,
                 options: Optional[Dict] = None,
                 saver: Optional[object] = None): # Accepte une instance de Saver
        """
        Initialise l'étape de superposition.

        Args:
            name (str): Nom de l'étape.
            process_function (Callable): Fonction qui traite une paire (overlay, background).
                                         Doit retourner {'image': Image.Image, 'label': str}.
            overlay_dirs (List): Dossier(s) contenant les images overlays.
            background_dirs (List): Dossier(s) contenant les images de fond.
            image_output_dirs (List): Dossier(s) où sauvegarder les images résultantes.
            label_output_dirs (List): Dossier(s) où sauvegarder les labels YOLO (.txt).
            background_extensions (Tuple): Extensions pour les fichiers de fond.
            overlay_extensions (Tuple): Extensions pour les fichiers overlays.
            output_prefix (str): Préfixe à ajouter aux noms de fichiers générés.
            root_dir (Optional): Dossier racine pour résoudre les chemins relatifs.
            options (Optional[Dict]): Options à passer à process_function.
            saver (Optional[object]): Instance d'un ResultSaver (ou compatible) pour gérer la sauvegarde.
        """
        # Appel à l'init parent. On passe overlay_dirs comme "input" principal
        # et image_output_dirs comme "output" principal pour la résolution de chemin de base.
        # Les chemins spécifiques seront gérés par cette sous-classe.
        super().__init__(
            name=name,
            process_function=process_function,
            input_dirs=overlay_dirs,
            output_dirs=image_output_dirs, # Principalement pour que _resolve_paths fonctionne
            root_dir=root_dir,
            options=options
            # Pas besoin de passer fixed_input ici car on redéfinit run
        )

        # Stocker les chemins spécifiques résolus
        self.overlay_paths = self.input_paths # Réutilise l'attribut résolu par le parent
        self.background_paths = self._resolve_paths(background_dirs)
        self.image_output_paths = self.output_paths # Réutilise l'attribut résolu par le parent
        self.label_output_paths = self._resolve_paths(label_output_dirs)

        # Vérifier qu'on a bien les dossiers nécessaires
        if not self.overlay_paths: raise ValueError("overlay_dirs est requis.")
        if not self.background_paths: raise ValueError("background_dirs est requis.")
        if not self.image_output_paths: raise ValueError("image_output_dirs est requis.")
        if not self.label_output_paths: raise ValueError("label_output_dirs est requis.")

        self.background_extensions = background_extensions
        self.overlay_extensions = overlay_extensions
        self.output_prefix = output_prefix

        # Utiliser le saver fourni ou un saver par défaut si disponible
        # Cette partie dépend de comment ResultSaver est géré (instance unique ou par étape)
        self.saver = saver # if saver else ResultSaver() # Assigner le saver

    def run(self):
        """
        Exécute l'étape de superposition et de génération de labels.
        Liste les overlays et backgrounds une seule fois, puis itère.
        """
        self.processed_files_map = {} # input_overlay -> [output_image, output_label]
        print(f"--- Exécution Étape Spécialisée : {self.name} ---")
        overlay_input_dir = self.overlay_paths[0] # Prend le premier dossier d'overlay
        background_input_dir = self.background_paths[0] # Prend le premier dossier de fond
        image_output_dir = self.image_output_paths[0] # Prend le premier dossier de sortie image
        label_output_dir = self.label_output_paths[0] # Prend le premier dossier de sortie label

        print(f"Source Overlays : {overlay_input_dir}")
        print(f"Source Backgrounds: {background_input_dir}")
        print(f"Sortie Images   : {image_output_dir}")
        print(f"Sortie Labels   : {label_output_dir}")

        # Créer les dossiers de sortie
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Lister les fichiers une seule fois ---
        try:
            print("Listage des fichiers overlays...")
            overlay_files = sorted([
                f for f in overlay_input_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.overlay_extensions
            ])
            print(f"Trouvé {len(overlay_files)} overlays.")

            print("Listage des fichiers backgrounds...")
            background_files = sorted([
                f for f in background_input_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.background_extensions
            ])
            print(f"Trouvé {len(background_files)} backgrounds.")

        except FileNotFoundError as e:
            print(f"Erreur [{self.name}]: Dossier d'entrée non trouvé: {e}. Étape annulée.")
            return
        except Exception as e:
            print(f"Erreur [{self.name}]: Erreur lors du listage des fichiers: {e}. Étape annulée.")
            return

        if not overlay_files:
            print(f"Avertissement [{self.name}]: Aucun fichier overlay trouvé dans {overlay_input_dir}. Étape terminée.")
            return
        if not background_files:
            print(f"Avertissement [{self.name}]: Aucun fichier background trouvé dans {background_input_dir}. Étape terminée.")
            return

        # --- Boucle de traitement ---
        num_overlays = len(overlay_files)
        num_backgrounds = len(background_files)
        print(f"Traitement de {num_overlays} overlays...")

        processed_count = 0
        errors_count = 0

        for i, overlay_path in enumerate(overlay_files):
            # Sélectionner un background (logique de boucle comme dans le script original)
            background_path = background_files[i % num_backgrounds]

            try:
                # Appeler la fonction de traitement fournie
                result = self.process_function(
                    overlay_path,
                    background_path,
                    **self.process_kwargs # Passe les options (ex: yolo_class_id, min/max_scale)
                )

                if result and isinstance(result, dict) and 'image' in result and 'label' in result:
                    # --- Sauvegarde (déléguée au Saver si possible) ---
                    image_data = result['image']
                    label_data = result['label']
                    # Nom basé sur l'overlay, avec préfixe
                    base_name = f"{self.output_prefix}{overlay_path.stem}"
                    img_output_path = image_output_dir / f"{base_name}.jpg" # Force JPG
                    label_output_path = label_output_dir / f"{base_name}.txt"

                    saved_paths = []
                    if self.saver:
                         # Le Saver doit être capable de gérer ce cas spécifique
                         # On pourrait lui passer le dict entier ou les éléments séparément
                         # Ici on suppose qu'il gère le dict comme avant, mais on vérifie les sorties
                         temp_result_for_saver = {
                             'image': image_data,
                             'label': label_data,
                             'overlay_name': overlay_path.stem # Le saver en a besoin pour le nommage
                         }
                         # Important: Le saver doit utiliser les bons output_dirs.
                         # On lui passe explicitement les dossiers pour cette étape.
                         saved_paths = self.saver.save(
                             temp_result_for_saver,
                             overlay_path, # L'overlay est l'input "primaire" pour le mapping
                             [image_output_dir, label_output_dir], # Les deux sorties attendues
                             self.name
                         )
                    else:
                         # Sauvegarde directe si pas de Saver (moins recommandé)
                         print(f"Avertissement [{self.name}]: Aucun Saver fourni. Sauvegarde directe.")
                         try:
                             image_data.save(img_output_path, format='JPEG')
                             saved_paths.append(img_output_path)
                             with open(label_output_path, 'w', encoding='utf-8') as f:
                                 f.write(label_data)
                             saved_paths.append(label_output_path)
                         except Exception as e_save:
                             print(f"Erreur [{self.name}]: Échec sauvegarde directe pour {base_name}: {e_save}")

                    if len(saved_paths) == 2: # Si les deux fichiers ont été créés
                        self.processed_files_map[overlay_path] = saved_paths
                        processed_count += 1
                    else:
                        errors_count += 1 # Erreur pendant la sauvegarde

                elif result is None:
                    # La fonction de traitement a signalé un échec géré
                    errors_count += 1
                    # print(f"Info [{self.name}]: Traitement ignoré pour {overlay_path.name} sur {background_path.name}")
                else:
                    # Retour inattendu de process_function
                    print(f"Erreur [{self.name}]: Retour inattendu de process_function pour {overlay_path.name}.")
                    errors_count += 1

            except Exception as e_proc:
                print(f"Erreur [{self.name}]: Échec critique lors du traitement de {overlay_path.name} sur {background_path.name}: {e_proc}")
                errors_count += 1
                # import traceback
                # traceback.print_exc()

            # Affichage de la progression (simple)
            if (i + 1) % 100 == 0 or (i + 1) == num_overlays:
                 print(f"  Progression : {i + 1}/{num_overlays}")


        print(f"--- Étape {self.name} terminée ---")
        print(f"  {processed_count} paires traitées et sauvegardées avec succès.")
        if errors_count > 0:
            print(f"  {errors_count} erreurs rencontrées.")