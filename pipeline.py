from typing import Callable, Union, List
from pathlib import Path


class ProcessingStep:
    def __init__(self,
                 name: str,
                 process_fn: Callable,
                 input_dirs: Union[str, List[str]],
                 output_dirs: Union[str, List[str]]):
        self.name = name
        self.process_fn = process_fn
        self.input_dirs = input_dirs if isinstance(input_dirs, list) else [input_dirs]
        self.output_dirs = output_dirs if isinstance(output_dirs, list) else [output_dirs]

    def run(self):
        # Exemple simple : un input -> un output
        input_path = Path(self.input_dirs[0])
        output_path = Path(self.output_dirs[0])
        output_path.mkdir(exist_ok=True)

        for file in input_path.glob("*"):
            result = self.process_fn(file)
            output_file = output_path / file.name
            result.save(output_file)  # selon le type d'objet retourné


class ProcessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step: ProcessingStep):
        self.steps.append(step)

    def run(self, from_step=0):
        for i, step in enumerate(self.steps[from_step:], start=from_step):
            print(f"Running step {i}: {step.name}")
            step.run()
