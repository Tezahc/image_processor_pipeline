from typing import Callable, Union, List
from pathlib import Path


class ProcessingStep:
    def __init__(self,
                 name: str,
                 process_fn: Callable,
                 output_dir: str,
                 input_dir: str = None):
        self.name = name
        self.process_fn = process_fn
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self, input_dir=None):
        input_path = Path(input_dir or self.input_dir)
        output_path = Path(self.output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        for file in input_path.glob("*"):
            result = self.process_fn(file)
            result.save(output_path / file.name)


class MultiInputOutputStep(ProcessingStep):
    def run(self):
        input_path1 = Path(self.input_dirs[0])
        input_path2 = Path(self.input_dirs[1])
        output_path1 = Path(self.output_dirs[0])
        output_path2 = Path(self.output_dirs[1])

        output_path1.mkdir(exist_ok=True)
        output_path2.mkdir(exist_ok=True)

        for file1, file2 in zip(input_path1.glob("*"), input_path2.glob("*")):
            res1, res2 = self.process_fn(file1, file2)
            res1.save(output_path1 / file1.name)
            res2.save(output_path2 / file2.name)


class ProcessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step: ProcessingStep):
        if not self.steps:
            # C'est la toute première étape
            if step.input_dir is None:
                raise ValueError("The first step must have an input_dir defined.")
        else:
            # On a déjà au moins un step précédent
            previous_step = self.steps[-1]
            if step.input_dir is None:
                # Si pas défini, on prend automatiquement l'output du précédent
                step.input_dir = previous_step.output_dir

        self.steps.append(step)

    def run(self, from_step=0, only_one=False):
        for i, step in enumerate(
            [self.steps[from_step]] if only_one else self.steps[from_step:],
            start=from_step,
        ):
            print(f"Running step {i}: {step.name}")
            step.run()
