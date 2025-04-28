from pathlib import Path
from typing import Callable, List, Dict, Optional, Union, Tuple, Iterator, Literal
from tqdm import tqdm


class ProcessingStep:
    def __init__(self,
                 name: str,
                 process_function: Callable,
                 input_dirs: List[Union[str, Path]],
                 output_dirs: List[Union[str, Path]],
                 pairing_strategy: Literal['one_input', 'zip', 'modulo', 'custom'] = 'one_input',
                 pairing_function: Optional[Callable[[List[List[Path]]], Iterator[Tuple]]] = None,
                 fixed_input: bool = False,
                 root_dir: Optional[Union[str, Path]] = None,
                 options: Optional[Dict] = None):
        """
        Initialise une étape de traitement générique.

        Args:
            name (str): Nom lisible de l'étape.
            process_function (Callable): La fonction qui effectue le traitement.
                Signature attendue : (*input_paths: Path, output_paths: List[Path], **options) -> Optional[Union[Path, List[Path]]]
                Doit accepter un nombre variable d'arguments Path en entrée (selon la stratégie),
                la liste des chemins de sortie, et les options.
                Doit retourner le(s) chemin(s) du/des fichier(s) sauvegardé(s), ou None si échec/rien à sauver.
            input_dirs (List): Liste des chemins des dossiers d'entrée (relatifs ou absolus).
            output_dirs (List): Liste des chemins des dossiers de sortie (relatifs ou absolus).
            pairing_strategy (PairingStrategy): Comment combiner les fichiers des input_dirs.
                Options: 'one_input' (défaut), 'zip', 'product', 'modulo', 'custom'.
            pairing_function (Callable): Requis si strategy='custom'. Voir doc _generate_processing_args.
            fixed_input (Bool): TODO: ajouter description déjà écrite ailleurs...
            root_dir (Optional): Dossier racine pour résoudre les chemins relatifs.
            options (Optional[Dict]): Arguments (kwargs) additionnels passés à process_function.
        """
        self.name = name
        self.process_function = process_function
        self.root_dir = Path(root_dir) if root_dir else Path('.')
        self.process_kwargs = options or {}

        # Résolution des chemins
        self.input_paths: List[Path] = self.resolve_paths(input_dirs or [])
        self.output_paths: List[Path] = self.resolve_paths(output_dirs)
        self.fixed_input = fixed_input

        if not self.output_paths:
            raise ValueError(f"L'étape '{self.name}' doit avoir au moins un 'output_dirs' défini.")

        # Validation de la stratégie (sécurité runtime)
        # Note: Le type Literal fait déjà une vérification statique
        valid_strategies = ['one_input', 'zip', 'modulo', 'custom']
        if pairing_strategy not in valid_strategies: # Vérifie si la valeur est bien une des littérales
            raise ValueError(f"Stratégie de pairing '{pairing_strategy}' invalide. Choisir parmi: {valid_strategies}")
        if pairing_strategy == 'custom' and not callable(pairing_function):
            raise ValueError("Une `pairing_function` valide est requise pour la stratégie 'custom'.")
        self.pairing_strategy = pairing_strategy
        self.pairing_function = pairing_function

        # Map pour suivre les sorties générées par entrée(s)
        self.processed_files_map: Dict[Tuple[Path, ...], Union[Path, List[Path]]] = {} # Clé est tuple de Path d'entrée

    def resolve_paths(self, dir_list: List[Union[str, Path]]) -> List[Path]:
        """Convertit et résout les chemins par rapport au root_dir."""
        resolved = []
        for d in dir_list:
            path = Path(d)
            # Si le chemin n'est pas absolu, on le considère relatif au root_dir
            if not path.is_absolute():
                resolved.append(self.root_dir / path)
            else:
                resolved.append(path)
        return resolved

    # TODO: Implémenter __str__ pour un résumé utile de l'étape (inputs, outputs, stratégie)
    def __str__(self) -> str:
        input_str = ", ".join([p.name for p in self.input_paths])
        output_str = ", ".join([p.name for p in self.output_paths])
        return (f"Étape '{self.name}':\n"
                f"  Entrée(s) : [{input_str}] (Stratégie: {self.pairing_strategy})\n"
                f"  Sortie(s) : [{output_str}]\n"
                f"  Options   : {self.process_kwargs}")

    def _get_files_from_inputs(self) -> List[List[Path]]:
        """Liste les fichiers de chaque dossier d'entrée. Lève une erreur si un dossier n'existe pas."""
        all_file_lists = []
        if not self.input_paths:
            print(f"Avertissement [{self.name}]: Aucun dossier d'entrée défini.")
            return [] # Retourne une liste vide de listes

        print(f"Info [{self.name}]: Listage des fichiers d'entrée...")
        for i, input_dir in enumerate(self.input_paths):
            if not input_dir.is_dir():
                # Lever une erreur si le dossier n'existe pas
                raise FileNotFoundError(f"Le dossier d'entrée spécifié n'existe pas: '{input_dir}' pour l'étape '{self.name}'")

            try:
                # Lister tous les fichiers (pas de filtrage d'extension ici) et trier
                files = sorted([f for f in input_dir.iterdir() if f.is_file()])
                print(f"  '{input_dir.name}': {len(files)} fichier(s) trouvé(s).")
                all_file_lists.append(files)
            except Exception as e:
                # Gérer autres erreurs potentielles (ex: permissions)
                print(f"Erreur [{self.name}]: Échec du listage de {input_dir}: {e}")
                # On pourrait lever une erreur ici aussi, ou juste ajouter une liste vide
                # Levons une erreur pour être strict
                raise IOError(f"Impossible de lister les fichiers dans {input_dir}") from e

        return all_file_lists

    def _generate_processing_inputs(self, input_file_lists: List[List[Path]]) -> Iterator[Tuple[Path, ...]]:
        """
        Génère les tuples d'arguments (chemins) pour process_function basé sur la stratégie.

        Args:
            input_file_lists: Liste contenant une liste de Path pour chaque dossier d'entrée.

        Yields:
            Tuple[Path, ...]: Un tuple de chemins (Path) à passer comme *args
                              à la fonction de traitement pour chaque appel.
        """
        input_count = len(input_file_lists)

        if self.pairing_strategy == 'one_input':
            if input_count == 0: # Sécurité
                raise ValueError("Stratégie 'one_input' mais aucun dossier d'entrée fourni.")
            list1 = input_file_lists[0]
            if not list1:
                raise ValueError(f"Stratégie 'one_input' mais le dossier d'entrée '{self.input_paths[0].name}' est vide.")

            for file_path in list1:
                yield (file_path,) # Tuple avec un seul élément

        elif self.pairing_strategy == 'zip':
            if input_count < 2:
                raise ValueError("La stratégie 'zip' requiert au moins 2 dossiers d'entrée.")
            # Vérifier qu'aucune liste n'est vide (zip s'arrêterait, mais c'est plus clair de prévenir)
            if not all(input_file_lists):
                empty_folders = [str(self.input_paths[i]) for i, lst in enumerate(input_file_lists) if not lst]
                raise ValueError(f"Stratégie 'zip' requiert des fichiers dans tous les dossiers d'entrée. Dossiers vides: {empty_folders}")

            yield from zip(*input_file_lists)

        elif self.pairing_strategy == 'modulo':
            if input_count != 2:
                raise ValueError("La stratégie 'modulo' requiert exactement 2 dossiers d'entrée.")
            list1 = input_file_lists[0]
            list2 = input_file_lists[1]
            if not list1 or not list2:
                empty_folders = []
                if not list1: empty_folders.append(str(self.input_paths[0]))
                if not list2: empty_folders.append(str(self.input_paths[1]))
                raise ValueError(f"Stratégie 'modulo' requiert des fichiers dans les deux dossiers. Dossiers vides: {empty_folders}")

            num_list2 = len(list2)
            for i, path1 in enumerate(list1):
                path2 = list2[i % num_list2]
                yield (path1, path2)

        elif self.pairing_strategy == 'custom':
            if self.pairing_function:
                yield from self.pairing_function(input_file_lists)
            else:
                raise ValueError("Fonction `pairing_function` manquante pour la stratégie 'custom'.")

        else:
            # Normalement impossible grâce à Literal et la vérif init
            raise NotImplementedError(f"Stratégie de pairing '{self.pairing_strategy}' non implémentée.")

    def run(self):
        """Exécute l'étape de traitement pour tous les éléments/paires d'entrée."""
        self.processed_files_map = {}
        print(f"--- Exécution Étape : {self.name} ---")
        # print(self) # Utiliser __str__ pour afficher les détails si besoin

        # Créer les dossiers de sortie (une seule fois au début)
        print(f"Info [{self.name}]: Vérification/Création des dossiers de sortie...")
        for output_path in self.output_paths:
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"  Sortie -> '{output_path}'")
            except Exception as e:
                raise IOError(f"Impossible de créer le dossier de sortie '{output_path}': {e}") from e

        # 1. Lister les fichiers d'entrée
        try:
            input_file_lists = self._get_files_from_inputs()
            # Vérification globale : au moins un fichier dans au moins un dossier ?
            if not all(input_file_lists):
                # Au moins une liste ne contient pas de fichier
                raise FileNotFoundError(f"Aucun fichier trouvé dans les dossiers d'entrée {[str(p) for p in self.input_paths]} pour l'étape '{self.name}'.")
        except (FileNotFoundError, ValueError, IOError) as e:
            # Erreur lors du listage ou dossier vide alors que requis
            print(f"Erreur [{self.name}]: Condition préalable non remplie pour démarrer l'étape. {e}")
            # On arrête l'étape ici
            return

        # 2. Obtenir l'itérateur d'arguments
        try:
            argument_iterator = self._generate_processing_inputs(input_file_lists)
        except (ValueError, NotImplementedError) as e:
            print(f"Erreur [{self.name}]: Impossible de générer les arguments pour la stratégie '{self.pairing_strategy}'. {e}")
            return  # Arrêter l'étape

        # 3. Boucle de traitement avec tqdm
        processed_count = 0
        errors_count = 0
        print(f"Info [{self.name}]: Démarrage du traitement...")

        # Utilisation de tqdm pour la barre de progression
        progress_bar = tqdm(argument_iterator, desc=self.name, unit="item", smoothing=0)
        for input_args_tuple in progress_bar:
            try:
                # Clé pour le suivi
                input_key = input_args_tuple

                # Appel de la fonction de traitement
                # Elle reçoit les chemins d'entrée, les chemins de sortie, et les options
                saved_output_paths: Optional[Union[Path, List[Path]]] = self.process_function(
                    *input_args_tuple,              # Dépaquette les chemins d'entrée
                    output_paths=self.output_paths, # Passe la liste des dossiers de sortie
                    **self.process_kwargs           # Passe les options définies pour l'étape
                )

                # Vérifier le retour de la fonction de traitement
                if saved_output_paths:
                    # S'assurer que c'est bien Path ou List[Path] (pourrait être plus strict)
                    if isinstance(saved_output_paths, Path) or \
                       (isinstance(saved_output_paths, list) and all(isinstance(p, Path) for p in saved_output_paths)):
                        self.processed_files_map[input_key] = saved_output_paths
                        processed_count += 1
                    else:
                        print(f"Avertissement [{self.name}]: Retour invalide de process_function pour {input_key} (type: {type(saved_output_paths)}). Attendu Path, List[Path] ou None.")
                        errors_count += 1
                else:
                    # La fonction a retourné None (échec géré ou rien à sauvegarder)
                    # On peut choisir de le compter comme une erreur ou non. Comptons-le.
                    errors_count += 1
                    # Optionnel: logger l'entrée qui n'a rien produit
                    # print(f"Debug [{self.name}]: Aucun fichier sauvegardé pour l'entrée {input_key}")

            except Exception as e_proc:
                # Erreur inattendue DANS process_function ou lors de son appel
                print(f"\nErreur [{self.name}]: Échec critique lors du traitement de {input_args_tuple}: {e_proc}")
                # Afficher le traceback peut être utile pour le débogage
                # import traceback
                # traceback.print_exc()
                errors_count += 1
                # Optionnel: mettre à jour la description de tqdm
                # progress_bar.set_postfix_str(f"Erreur sur {input_args_tuple[0].name}", refresh=True)

        progress_bar.close() # Fermer proprement la barre tqdm

        print(f"--- Étape {self.name} terminée ---")
        print(f"  {processed_count} élément(s) traité(s) avec succès (fichier(s) de sortie généré(s)).")
        if errors_count > 0:
            print(f"  {errors_count} erreur(s) ou traitement(s) sans sortie.")


class ProcessingPipeline:
    def __init__(self, root_dir: Optional[str] = None):
        self.steps: List[ProcessingStep] = []
        # définit le dossier source du pipeline → obligatoire ?
        self.root_dir = Path(root_dir) if root_dir else None

    def add_step(self, step: ProcessingStep, position=None):
        # Vérification : la première étape doit avoir des inputs définis
        if not self.steps and step.input_paths is None:
            raise ValueError(f"The first step ('{step.name}') must have input_dir defined.")
            # self.steps.append(step)
            # return

        # si un dossier racine est défini dans le pipeline, mais pas dans l'étape,
        # il est transmis à l'étape dès son ajout
        if self.root_dir and not step.root_dir:
            step.root_dir = self.root_dir
            # modifie les dossiers d'input/output s'ils sont définis comme des noms de dossier ou des path relatifs

            # TODO: Tester ce check. Peut être pas ici puisqu'on modifie input_dir juste après
            step.input_paths = step.resolve_paths(step.input_paths)
            step.output_paths = step.resolve_paths(step.output_paths)

        if position is None or position < 0:
            previous_step = self.steps[-1] if self.steps else None

            if step.input_paths is None:
                if previous_step is None:
                    raise ValueError("The first step must have an input_paths defined.")
                step.input_paths = previous_step.output_paths

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
                step.input_dir = previous_step.output_paths

            # Insertion
            self.steps.insert(position, step)

            # Mettre à jour input_dir de la prochaine étape si elle n'est pas fixe
            if next_step and not next_step.fixed_input:
                next_step.input_dir = step.output_paths

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