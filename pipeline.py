from typing import Callable, Optional, Union, List
from pathlib import Path
import cv2
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
                    cv2.imwrite(str(output_file), result) # TODO: faudra gérer l'enregistrement si la fonction utilise une autre lib (ex PIL) -> méthode save avec un case ?
                    self.processed_files.append(output_file)
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
