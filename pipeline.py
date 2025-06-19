import concurrent
import concurrent.futures
from os import cpu_count
from pathlib import Path
from typing import Callable, List, Dict, Optional, Tuple, Iterator, Literal
from warnings import warn
from tqdm.notebook import tqdm
import random

MODES = ('one_input', 'zip', 'modulo', 'custom')


class ProcessingStep:
    def __init__(self,
                 name: str,
                 process_function: Callable,
                 input_dirs: Optional[str | Path | List[str | Path]] = None,
                 output_dirs: Optional[str | Path | List[str | Path]] = None,
                 pairing_method: Literal[*MODES] = 'one_input',  # type: ignore
                 pairing_function: Optional[Callable[[List[List[Path]]], Iterator[Tuple]]] = None,
                 fixed_input: bool = False,
                 root_dir: Optional[str | Path] = None,
                 sample_k: Optional[int] = None,
                 workers: Optional[int] = 1,
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
        # TODO: accepter le nom d'une étape lors des manipulations (insertions, ...)
        self.name = name 
        self.process_function = process_function
        # TODO : Tester None au lieu de Path('.') ou même cwd() probablement encore mieux
        self.root_dir = Path(root_dir) if root_dir else None 
        self.process_kwargs = options or {}
        self.sample_k = sample_k

        # Résolution des chemins
        self.input_paths: List[Path] = self._resolve_paths(input_dirs or [])
        self.output_paths: List[Path] = self._resolve_paths(output_dirs or [])
        # TODO: si output_dir non rempli, créer un dossier du même nom que l'étape ? 
        # => vérification d'accents et replace les ` ` par `_`
        self.fixed_input = fixed_input

        if not self.output_paths:
            raise ValueError(f"L'étape '{self.name}' doit avoir au moins un 'output_dirs' défini.") 
            # et bah pas forcément ! si on écrase le fichier d'input ! ->  on donne l'output = input donc pas vide ?
            # TODO: juste enlever ce check ou y'a d'autres implications ?

        # Validation du mode (sécurité runtime) -> késako ?
        # NOTE: Le type Literal fait déjà une vérification statique → statique = ? ouvrir un dico...
        if pairing_method not in set(MODES): 
            raise ValueError(f"Mode d'appariement' '{pairing_method}' invalide. Choisir parmi: {MODES}")
        # On laisse pour avenir lointain
        if pairing_method == 'custom' and not callable(pairing_function):
            raise ValueError("Une `pairing_function` valide est requise pour le mode 'custom'.")
        self.pairing_method = pairing_method
        self.pairing_function = pairing_function

        # Map pour suivre les sorties générées par entrée(s)
        self.processed_files_map: Dict[Tuple[Path, ...], Path | List[Path]] = {}  # Clé est tuple de Path d'entrée

        # Gestion de la parallélisation
        nb_cpus = cpu_count()
        if workers > nb_cpus:
            warn(f"Nombre de workers parallèles ajusté à {nb_cpus} (maximum système).")
        if workers == -1:
            workers = nb_cpus
        self.parallels_workers = min(workers, nb_cpus)

    def _resolve_paths(self, dir_list: str | Path | List[str | Path]) -> List[Path]:
        """Convertit et résout les chemins par rapport au root_dir. 
        Chaque chemin de la liste est converti en Path.
        Si un chemin n'est pas absolu, il est considéré relatif au dossier racine.
        """
        # assert dans une liste
        dir_list = [dir_list] if not isinstance(dir_list, list) else dir_list

        resolved = []
        for folder in dir_list:
            if not isinstance(folder, (str, Path)):
                raise ValueError(f"un élément ne représente pas un dossier ou un chemin : {folder}")
            
            dir_path = Path(folder)  # Path(Path()) pose pas de problème
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
            
            # TODO: ce try serait mieux catch en amont et directement raise des erreur
            try:
                # Lister tous les fichiers et trier
                files = sorted([f for f in input_dir.iterdir() if f.is_file()])
                # TODO : fonction utilitaire pour gérer les pluriel. (ou lib "inflect")
                print(f"  '{input_dir.name}' : {len(files)} fichiers trouvés.") 
                all_file_lists.append(files)
            except Exception as e:
                # FIXME: horrible : exception (tout) renvoie une IOError
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
        
        # Prélève le nombre d'éléments prescrit. Aux même ids pour chaque liste d'input
        if self.sample_k and isinstance(self.sample_k, int):
            sample_ids = random.sample(range(len(input_file_lists[0])), self.sample_k)
            input_file_lists = [[file_list[i] for i in sample_ids] for file_list in input_file_lists]
        
        # Mode de génération :
        if self.pairing_method == 'one_input':
            if input_len == 0:  # Sécurité
                raise ValueError("Mode 'one_input' mais aucun dossier d'entrée fourni.")
            input_files = input_file_lists[0]

            for file_path in input_files:
                yield (file_path,)  # Tuple avec un seul élément

        elif self.pairing_method == 'zip':
            if input_len < 2:
                raise ValueError("Le mode 'zip' requiert au moins 2 dossiers d'entrée.")

            yield from zip(*input_file_lists)

        elif self.pairing_method == 'modulo':
            # à la différence de zip, modulo revient au début de la deuxième liste si elle est totalement parcourue
            if input_len != 2:
                raise ValueError("Le mode 'modulo' requiert exactement 2 dossiers d'entrée.")
            list1 = input_file_lists[0]
            list2 = input_file_lists[1]

            # TODO ? shuffle_input en option ? pareil pour zip ?
            # shuffle seulement la 2e liste (backgrounds) suffisant.
            random.shuffle(list2)

            list2_len = len(list2)
            for i, path1 in enumerate(list1):
                path2 = list2[i % list2_len]
                yield (path1, path2)
        
        elif self.pairing_method == 'sampling':
            # méthode spécifique pour l'étape de transformation
            # TODO: à généraliser ?

            input_files = input_file_lists[0]
            input_labels = input_file_lists[1]

            # Sample un set des fichiers où appliquer la transfo
            blur_sample = set(random.sample(input_files, int(len(input_files)*0.3)))
            # Crée une liste de booléens (donnés en paramètres d'input) 
            # Indiquant si l'élément évalué doit subir la transformation
            do_blur = [i in blur_sample for i in input_files]

            # idem pour la transfo RGB
            rgb_sample = set(random.sample(input_files, int(len(input_files)*0.3)))
            do_rgb = [i in rgb_sample for i in input_files]

            yield from zip(input_files, input_labels, do_blur, do_rgb)

        
        elif self.pairing_method == 'custom':
            if not self.pairing_function:
                raise ValueError("Fonction `pairing_function` manquante pour le mode 'custom'.")
            
            yield from self.pairing_function(input_file_lists)

        else:
            # Normalement impossible grâce à Literal et la vérif init
            raise NotImplementedError(f"Mode d'appariement' '{self.pairing_method}' non implémentée.")

    def run(self):
        """Exécute l'étape de traitement pour tous les éléments/paires d'entrée."""
        self.processed_files_map = {}  # Retrace les résultats, y'a un truc à faire avec... un jour...
        print(f"--- Exécution Étape : {self.name} ---")
        # print(self) # Utiliser __str__ pour afficher les détails si besoin 
        # TODO : paramètre verbose

        # 1. Créer les dossiers de sortie (une seule fois au début)
        print(f"Info [{self.name}]: Vérification/Création des dossiers de sortie...")
        for output_path in self.output_paths:
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"  Sortie -> '{output_path}'")
            except IOError as ioe:
                raise IOError(f"Impossible de créer le dossier de sortie '{output_path}': {ioe}") from ioe 
            except Exception as e:
                print(f"Erreur lors de la création du dossier {output_path}. {e}")
                return
        
        # 2. Lister les fichiers d'entrée
        try:
            input_file_lists = self._get_files_from_inputs()
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"Erreur [{self.name}]: Condition préalable non remplie pour démarrer l'étape. {e}")
            return

        # 3. Obtenir l'itérateur d'arguments
        try:
            argument_iterator = self._generate_processing_inputs(input_file_lists)
            # TODO et si le mode du générateur renvoyait un tuple avec le total d'opération ? (pout tqdm tsé 👀) 
            # => solution : créer une classe générator qui yield comme la méthode et possède un attribut .total
        except (ValueError, NotImplementedError) as e:
            print(f"Erreur [{self.name}]: Impossible de générer les arguments pour le mode '{self.pairing_method}'. {e}")
            return

        # Calcul du total pour tqdm
        total_items = None
        try:
            if self.pairing_method == 'one_input': total_items = len(input_file_lists[0])
            elif self.pairing_method == 'modulo': total_items = len(input_file_lists[0])
            elif self.pairing_method == 'zip': total_items = min(len(lst) for lst in input_file_lists if lst)
            else: raise ValueError(f"mode d'appariemment inconnu, utiliser un parmi {MODES}")
        except Exception: total_items = None

        # --------------------------------------------------------------------------------------------
        #                    4. Boucle de traitement (avec tqdm SoonTM tkt)
        # --------------------------------------------------------------------------------------------

        processed_count, errors_count = self._processing_loop(argument_iterator, total_items)

        # TODO: intégrer le timings (quoique, avec tqdm.... :pray:)
        print(f"--- Étape {self.name} terminée ---") 
        print(f"  {processed_count} éléments traités avec succès (fichiers de sortie générés).")
        if errors_count > 0:
            print(f"  {errors_count} erreur(s) ou traitement(s) sans retour.")

    def _processing_loop(self, 
                         argument_iterator: Iterator[Tuple[Path, ...]], 
                         total_items: int) -> Tuple[int, int]:
        """Exécute la boucle de traitement principale, soit en séquentiel, soit en parallèle.
        Met à jour self.processed_files_map et retourne les compteurs.
        """
        processed_count = 0
        error_count = 0

        # --- Logique Séquentielle ---
        if not self.parallels_workers or 0 <= self.parallels_workers <= 1:
            print(f"Info [{self.name}]: Exécution en mode séquentiel...")
            for input_args_tuple in tqdm(argument_iterator, 
                                         desc=self.name, 
                                         total=total_items, 
                                         unit="item", 
                                         leave=True, 
                                         smoothing=0):
                try:
                    # Clé pour le suivi
                    input_key = input_args_tuple

                    # Appel de la fonction de traitement
                    saved_output_paths: Optional[Path | List[Path]] = self.process_function(
                        *input_args_tuple,              # Dépaquette les chemins d'entrée
                        output_dirs=self.output_paths,  # liste des dossiers de sortie
                        **self.process_kwargs           # Passage des options en kwargs
                    )

                    # Vérification du retour de la fonction de traitement
                    if saved_output_paths:
                        # on vérifie qu'on a bien un type Path (ou liste de paths)
                        # utilité ? 
                        if isinstance(saved_output_paths, Path) or \
                           (isinstance(saved_output_paths, list) and all(isinstance(p, Path) for p in saved_output_paths)):
                            self.processed_files_map[input_key] = saved_output_paths
                            processed_count += 1
                        else:
                            # TODO: utiliser des vrais warn ou système de logging
                            tqdm.write(f"Avertissement [{self.name}]: Retour invalide de {self.process_function} pour {input_key} (type: {type(saved_output_paths)}).")
                            error_count += 1
                    else:
                        # la fonction a retourné None (échec géré ou rien à sauvegarder)
                        error_count += 1
                        # Option : logger l'entrée qui n'a rien produit
                except Exception as e_proc:
                    # Erreur innatendue dans process_function ou lors de l'appel
                    tqdm.write(f"\nErreur [{self.name}]: Échec traitement de {input_args_tuple}: {e_proc}")
                    error_count += 1
            return processed_count, error_count
        
        # --- Logique Parallèle --- 
        elif self.parallels_workers > 1 or self.parallels_workers == -1:
            print(f"Info [{self.name}]: Exécution en mode parallèle avec {self.parallels_workers} workers...")

            list_of_input_args = list(argument_iterator)
            if not list_of_input_args:
                raise RuntimeError(f"Aucun argument à traiter après génération. Fin.")
            
            # Mettre à jour total_item si l'itérateur a été consomé
            if total_items is None:
                total_items = len(list_of_input_args)
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallels_workers) as executor:
                # Dictionnaire pour mapper les futures aux arguments d'entrée (pour le logging d'erreur)
                future_to_args: Dict[concurrent.futures.Future, Tuple[Path, ...]] = {}

                print(f"Info [{self.name}]: Soumission de {len(list_of_input_args)} tâches au pool de processus...")
                for input_args_tuple in list_of_input_args:
                    try:
                        future = executor.submit(
                            self.process_function,
                            *input_args_tuple,
                            output_dirs=self.output_paths,
                            **self.process_kwargs
                        )
                        future_to_args[future] = input_args_tuple
                    except Exception as e_submit:
                        tqdm.write(f"Erreur [{self.name}]: Échec de la soumission de la tâche pour {input_args_tuple}: {e_submit}")
                        error_count += 1
                
                for future in tqdm(concurrent.futures.as_completed(future_to_args.keys()),
                                   total=len(future_to_args),
                                   desc=self.name,
                                   unit="item",
                                   leave=True,
                                   smoothing=0):
                    # Récupère les args d'origine pour ce future
                    input_key = future_to_args[future] 

                    try:
                        saved_output_paths: Optional[Path | List[Path]] = future.result() # Bloque jusqu'à résultat

                        if saved_output_paths:
                            if isinstance(saved_output_paths, Path) or \
                               (isinstance(saved_output_paths, list) and all(isinstance(p, Path) for p in saved_output_paths)):
                                self.processed_files_map[input_key] = saved_output_paths
                                processed_count += 1
                            else:
                                tqdm.write(f"Avertissement [{self.name}]: Retour invalide (parallèle) pour {input_key} (type: {type(saved_output_paths)}).")
                                errors_count += 1
                        else:
                            errors_count += 1
                    except Exception as e_exec: # Erreur DANS le processus enfant
                        tqdm.write(f"\nErreur [{self.name}]: Échec tâche parallèle pour {input_key}: {e_exec}")
                        # import traceback; tqdm.write(traceback.format_exc()) # Pour debug
                        errors_count += 1
            return processed_count, error_count
        
        else:
            raise ValueError(f"")
            

class ProcessingPipeline:
    def __init__(self, root_dir: Optional[str | Path] = None):
        self.steps: List[ProcessingStep] = []
        # définit le dossier source du pipeline → obligatoire ?
        self.root_dir = Path(root_dir) if root_dir else None  # Path.cwd() ?

    def add_step(self, step: ProcessingStep, position=None):
        if not self.steps and not step.input_paths: 
            # step.input_paths n'est jamais None, au minimum []
            raise ValueError(f"La première étape ('{step.name}') doit avoir `input_dirs` définie.")

        # Si un dossier racine est défini dans le pipeline, mais pas dans l'étape :
        # il est transmis à l'étape dès son ajout sans l'écraser
        if self.root_dir and not step.root_dir:
            step.root_dir = self.root_dir
            # réévalue les chemins suite à la modification (éventuelle) du root_dir
            step.input_paths = step._resolve_paths(step.input_paths)
            step.output_paths = step._resolve_paths(step.output_paths)

        # Si pas précisé, on assert position à la dernière étape (volontairement = len(steps) donc out of index)
        position = len(self.steps) if position is None else position
        
        # Si l'étape ajoutée n'a pas d'input, on a forcément au moins une étape dans le pipeline,
        # on veut chainer les outputs précédents dans l'input actuel
        if not step.input_paths:
            if position == 0:
                raise IndexError(f"Insertion en première position, impossible de déduire les dossiers d'input pour {step.name}")
            
            # Tous les cas problématiques tombent dans un 'out of index' ou sont exclus par le check pos==0
            try:
                previous_step: ProcessingStep = self.steps[position - 1]
                # tant que l'étape est ajoutée à un autre point que la fin, on aura forcément une étape suivante
                next_step: ProcessingStep = self.steps[position] if position < len(self.steps) else None

                # === CHAINAGE ===
                # si on arrive ici, toutes les exceptions sont déjà élevées. Pas de risque de modification du pipeline malgré un blocage ensuite
                step.input_paths = previous_step.output_paths
                if next_step and not next_step.fixed_input:
                    # TODO : gérer les fixed_input en cas de len(input) > 1
                    next_step.input_paths = step.output_paths

            except IndexError as idx:
                raise ValueError(f"Position d'insertion invalide pour déduire les dossiers d'input de {step.name}.") from idx
            except Exception as e:
                raise RuntimeError(f"Erreur inattendue pour l'ajout de {step.name}") from e
        
        else:
            # on rentre si c'est la première étape (input_dirs *est* défini)
            pass

        # === AJOUT DE L'ÉTAPE ===
        self.steps.insert(position, step)

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
