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
        self.steps.append(step)

    def run(self, from_step=0):
        previous_output = None

        for i, step in enumerate(self.steps[from_step:], start=from_step):
            print(f"Running step {i}: {step.name}")

            if step.input_dir is None and previous_output is not None:
                step_input = previous_output
            else:
                step_input = step.input_dir

            step.run(input_dir=step_input)
            previous_output = step.output_dir
