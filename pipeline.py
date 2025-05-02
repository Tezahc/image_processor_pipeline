from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple, Iterator, Literal
from warnings import warn
from tqdm.notebook import tqdm
import random


MODES = Literal['one_input', 'zip', 'modulo', 'custom']
class ProcessingStep:
    def __init__(self,
                 name: str,
                 process_function: Callable,
                 input_dirs: Optional[str | Path | List[str | Path]] = None,
                 output_dirs: Optional[str | Path | List[str | Path]] = None,
                 pairing_method: MODES = 'one_input',
                 pairing_function: Optional[Callable[[List[List[Path]]], Iterator[Tuple]]] = None,
                 fixed_input: bool = False,
                 root_dir: Optional[str | Path] = None,
                 options: Optional[Dict] = None):
        """
        TODO: rewrite et uniformiser les styles de docstring (numpy ou Google)
        Initialise une étape de traitement générique.

        Args:
            name (str): Nom lisible de l'étape.
            process_function (Callable): La fonction qui effectue le traitement.
                Signature attendue : (*input_paths: Path, output_paths: List[Path], **options) -> Optional[Path | List[Path]]
                Doit accepter un nombre variable d'arguments Path en entrée (selon le mode),
                la liste des chemins de sortie, et les options.
                Doit retourner le(s) chemin(s) du/des fichier(s) sauvegardé(s), ou None si échec/rien à sauver.
            input_dirs (List): Liste des chemins des dossiers d'entrée (relatifs ou absolus).
            output_dirs (List): Liste des chemins des dossiers de sortie (relatifs ou absolus).
            pairing_method (PairingMethod): Comment combiner les fichiers des input_dirs.
                Options: 'one_input' (défaut), 'zip', 'modulo', 'custom'.
            pairing_function (Callable): Requis si method='custom'. Voir doc _generate_processing_args.
            fixed_input (Bool): TODO: ajouter description déjà écrite ailleurs...
                                TODO 2 : prendre en charge le fixed_input avec les listes de dossiers -> liste de bool ? oO
            root_dir (Optional): Dossier racine pour résoudre les chemins relatifs.
            options (Optional[Dict]): Arguments (kwargs) additionnels passés à process_function.
        """
        self.name = name # TODO: accepter le nom d'une étape lors des manipulations (insertions, ...)
        self.process_function = process_function
        self.root_dir = Path(root_dir) if root_dir else Path('.') # TODO : Tester None au lieu de Path('.') ou même cwd() probablement encore mieux
        self.process_kwargs = options or {}

        # Résolution des chemins
        self.input_paths: List[Path] = self._resolve_paths(input_dirs or [])
        self.output_paths: List[Path] = self._resolve_paths(output_dirs or [])
        # TODO: si output_dir non rempli, fallback sur input_dir (mais risque d'écraser les modifications) 
        # OU créer un dossier du même nom que l'étape ? => vérification d'accents et replace les ` ` par `_`
        self.fixed_input = fixed_input

        if not self.output_paths:
            raise ValueError(f"L'étape '{self.name}' doit avoir au moins un 'output_dirs' défini.") 
            # et bah pas forcément ! si on écrase le fichier d'input ! ->  on donne l'output = input donc pas vide ?
            # TODO: juste enlever ce check ou y'a d'autres implications ?

        # Validation du mode (sécurité runtime) -> késako ?
        # NOTE: Le type Literal fait déjà une vérification statique -> statique = ? ouvrir un dico...

        if pairing_method not in MODES: # Vérifie si la valeur est bien une des littérales
            raise ValueError(f"Mode d'appariement' '{pairing_method}' invalide. Choisir parmi: {MODES}")
        # On laisse pour avenir lointain
        if pairing_method == 'custom' and not callable(pairing_function):
            raise ValueError("Une `pairing_function` valide est requise pour le mode 'custom'.")
        self.pairing_method = pairing_method
        self.pairing_function = pairing_function

        # Map pour suivre les sorties générées par entrée(s)
        self.processed_files_map: Dict[Tuple[Path, ...], Path | List[Path]] = {} # Clé est tuple de Path d'entrée

    def _resolve_paths(self, dir_list: str | Path | List[str | Path]) -> List[Path]:
        """Convertit et résout les chemins par rapport au root_dir. 
        Chaque chemin de la liste est converti en Path.
        Si un chemin n'est pas absolu, il est considéré relatif au dossier racine.
        """
        # assert dans une liste
        dir_list = list(dir_list)

        resolved = []
        for folder in dir_list:
            if not isinstance(folder, (str, Path)):
                raise ValueError(f"un élément ne représente pas un dossier ou un chemin : {folder}")
            
            dir_path = Path(folder) # Path(Path()) pose pas de problème
            # Si le chemin n'est pas absolu, on le considère relatif au root_dir
            if not dir_path.is_absolute() and self.root_dir:
                resolved.append(self.root_dir / dir_path)
            else:
                resolved.append(dir_path)
        return resolved

    # TODO: Implémenter __str__ pour un résumé utile de l'étape (inputs, outputs, mode)
    def __str__(self) -> str:
        input_str = ", ".join([p.name for p in self.input_paths])
        output_str = ", ".join([p.name for p in self.output_paths])
        return (f"Étape '{self.name}':\n"
                f"  Entrée(s) : [{input_str}] (Mode: {self.pairing_method})\n"
                f"  Sortie(s) : [{output_str}]\n"
                f"  Options   : {self.process_kwargs}")

    def _get_files_from_inputs(self) -> List[List[Path]]:
        """Liste les fichiers de chaque dossier d'entrée. Lève une erreur si un dossier n'existe pas."""
        all_file_lists = []
        if not self.input_paths:
            raise ValueError(f"{self.name} : Aucun dossier d'entrée défini.")

        print(f"Info [{self.name}]: Récupération des chemins de fichiers d'entrée...")
        for input_dir in self.input_paths:
            if not input_dir.is_dir():
                # Lever une erreur si le dossier n'existe pas (ou n'est pas un dossier)
                raise FileNotFoundError(f"Le dossier d'entrée spécifié n'existe pas: '{input_dir}' pour l'étape '{self.name}'")

            try:
                # Lister tous les fichiers et trier
                files = sorted([f for f in input_dir.iterdir() if f.is_file()])
                # TODO : fonction utilitaire pour gérer les pluriel. (ou lib "inflect")
                print(f"  '{input_dir.name}' : {len(files)} fichiers trouvés.") 
                all_file_lists.append(files)
            except Exception as e:
                # Gérer autres erreurs potentielles (ex: permissions)
                raise IOError(f"Échec de l'inventaire du dossier {input_dir}") from e

        return all_file_lists

    def _generate_processing_inputs(self, input_file_lists: List[List[Path]]) -> Iterator[Tuple[Path, ...]]:
        """
        Génère les tuples d'arguments (chemins) pour `process_function` basé sur le mode.
        
        TODO : lister les modes disponibles avec courte description de la logique

        Args:
            input_file_lists: Liste contenant des listes de Path, pour chaque dossier d'entrée.

        Yields:
            Tuple[Path, ...]: Un tuple de chemins (Path) à passer comme *args
                              à la fonction de traitement pour chaque appel.
                              *chaque élément du tuple est "dépaqueté" et représente un argument dans la fonction attendue*
                              *ne peut pas être déclaré avec un nom d'argument*
                                def foo(*args): OK
                                def foo(bar=*args) Error
        """
        input_len = len(input_file_lists)

        # Vérifier qu'aucune liste n'est vide (zip s'arrêterait, mais c'est plus clair de prévenir)
        if not all(input_file_lists):
            empty_folders = [str(self.input_paths[i]) for i, lst in enumerate(input_file_lists) if not lst] 
            raise FileNotFoundError(f"Aucun fichier trouvé dans les dossiers d'entrée {empty_folders} pour l'étape '{self.name}'.")
        
        if self.pairing_method == 'one_input':
            if input_len == 0: # Sécurité
                raise ValueError("Mode 'one_input' mais aucun dossier d'entrée fourni.")
            input_files = input_file_lists[0]

            for file_path in input_files:
                yield (file_path,) # Tuple avec un seul élément

        elif self.pairing_method == 'zip':
            if input_len < 2:
                raise ValueError("Le mode 'zip' requiert au moins 2 dossiers d'entrée.")

            yield from zip(*input_file_lists)

        elif self.pairing_method == 'modulo':
            if input_len != 2:
                raise ValueError("Le mode 'modulo' requiert exactement 2 dossiers d'entrée.")
            list1 = input_file_lists[0]
            list2 = input_file_lists[1]

            # TODO ? shuffle_input en option ? pareil pour zip ?
            random.shuffle(list2)

            list2_len = len(list2)
            for i, path1 in enumerate(list1):
                path2 = list2[i % list2_len]
                yield (path1, path2)

        elif self.pairing_method == 'custom':
            if not self.pairing_function:
                raise ValueError("Fonction `pairing_function` manquante pour le mode 'custom'.")
            
            yield from self.pairing_function(input_file_lists)

        else:
            # Normalement impossible grâce à Literal et la vérif init
            raise NotImplementedError(f"Mode d'appariement' '{self.pairing_method}' non implémentée.")
    

    def run(self):
        """Exécute l'étape de traitement pour tous les éléments/paires d'entrée."""
        self.processed_files_map = {} # Trace les résultat, y'a un truc à faire avec... un jour...
        print(f"--- Exécution Étape : {self.name} ---")
        # print(self) # Utiliser __str__ pour afficher les détails si besoin 
        # TODO : paramètre verbose

        # 1. Créer les dossiers de sortie (une seule fois au début)
        print(f"Info [{self.name}]: Vérification/Création des dossiers de sortie...")
        for output_path in self.output_paths:
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"  Sortie -> '{output_path}'")
            except Exception as e:
                raise IOError(f"Impossible de créer le dossier de sortie '{output_path}': {e}") from e 
                # raise dans un except ?

        # 2. Lister les fichiers d'entrée
        try:
            input_file_lists = self._get_files_from_inputs()
        
        except (FileNotFoundError, ValueError, IOError) as e:
            # Erreur lors du listing ou dossier vide alors que requis
            print(f"Erreur [{self.name}]: Condition préalable non remplie pour démarrer l'étape. {e}")
            # On arrête l'étape ici
            return

        # 3. Obtenir l'itérateur d'arguments
        try:
            argument_iterator = self._generate_processing_inputs(input_file_lists)
            # TODO et si le mode du générateur renvoyait un tuple avec le total d'opération ? (pout tqdm tsé 👀) 
            # => solution : créer une classe générator qui yield comme la méthode et possède un attribut .total
        except (ValueError, NotImplementedError) as e:
            print(f"Erreur [{self.name}]: Impossible de générer les arguments pour le mode '{self.pairing_method}'. {e}")
            return  # Arrêter l'étape

        # --------------------------------------------------------------------------------------------
        #                    4. Boucle de traitement (avec tqdm SoonTM tkt)
        # --------------------------------------------------------------------------------------------
        processed_count = 0
        errors_count = 0 # TODO: quid d'un dictionnaires {processed:[], errors:[]} et on a le compte avec len() ?

        print(f"Info [{self.name}]: Démarrage du traitement...")
    
        # Utilisation de tqdm pour la barre de progression
        # progress_bar = tqdm(argument_iterator, desc=self.name, unit="item", total=total_items, smoothing=0)
        for input_args_tuple in argument_iterator:
            try:
                # Clé pour le suivi
                input_key = input_args_tuple
                
                # Appel de la fonction de traitement
                # Elle reçoit les chemins d'entrée, les chemins de sortie, et les options
                saved_output_paths: Optional[Path | List[Path]] = self.process_function(
                    *input_args_tuple,              # Dépaquette les chemins d'entrée
                    # TODO: dépaqueter aussi output_dirs (nécessite de revoir totu les fonctions...)
                    output_dirs=self.output_paths,  # Passe la liste des dossiers de sortie
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
                        # TODO : première occurence mais valable pour tous : Utiliser des vrais warnings (module spé) 
                        warn(f"Avertissement [{self.name}]: Retour invalide de process_function pour {input_key} (type: {type(saved_output_paths)}). Attendu Path, List[Path] ou None.")
                        errors_count += 1
                else:
                    # La fonction de traitement a retourné None (échec géré ou rien à sauvegarder)
                    # On peut choisir de le compter comme une erreur ou non. Comptons-le.
                    errors_count += 1
                    # Optionnel: logger l'entrée qui n'a rien produit
                    # print(f"Debug [{self.name}]: Aucun fichier sauvegardé (=> renvoyé) pour l'entrée {input_key}")

            except Exception as e_proc:
                # Erreur inattendue DANS process_function ou lors de son appel
                print(f"\nErreur [{self.name}]: Échec critique lors du traitement de {input_args_tuple}: {e_proc}")
                # Afficher le traceback peut être utile pour le débogage
                # import traceback
                # traceback.print_exc()
                errors_count += 1
                # Optionnel: mettre à jour la description de tqdm
                # progress_bar.set_postfix_str(f"Erreur sur {input_args_tuple[0].name}", refresh=True)

        # progress_bar.close() # Fermer proprement la barre tqdm

        print(f"--- Étape {self.name} terminée ---") # TODO: intégrer le timings (quoique, avec tqdm.... :pray:)
        print(f"  {processed_count} éléments traités avec succès (fichiers de sortie générés).")
        if errors_count > 0:
            print(f"  {errors_count} erreur(s) ou traitement(s) sans retour.")


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
        if self.root_dir: # and not step.root_dir:
            step.root_dir = self.root_dir
            # modifie les dossiers d'input/output s'ils sont définis comme des noms de dossier ou des path relatifs
            # TODO: gérer le fixed_input qui intervient sur la logique de chainage
            # TODO: Tester ce check. Peut être pas ici puisqu'on modifie input_dir juste après
            step.input_paths = step._resolve_paths(step.input_paths)
            step.output_paths = step._resolve_paths(step.output_paths)

        # Ajout de l'étape en dernière position (défaut)
        if position is None or position < 0:
            previous_step = self.steps[-1] if self.steps else None

            # chaînage des dossiers d'output deviennent l'input des suivant
            # TODO gérer avec les listes d'input/output => méthode dédiée
            if not step.input_paths:
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
        # TODO: vérifier si un seul des dossiers d'output des étapes à run n'est pas vide => ne run pas
        # évite les runs par accident
        # cette vérification ne sera pas faite sur les step.run() pour permettre d'écraser
        if from_step_index < 0 or from_step_index >= len(self.steps):
            raise IndexError(f"Invalid start index {from_step_index}. Pipeline has {len(self.steps)} steps.")
        
        steps_to_do = [self.steps[from_step_index]] if only_one else self.steps[from_step_index:]
        
        for i, step in enumerate(steps_to_do, start=from_step_index):
            print(f"Running étape {i}: {step.name}")
            step.run()
