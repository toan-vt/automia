from pathlib import Path
from dataclasses import dataclass
import yaml


@dataclass
class ExperimentConfig:
    context: str
    function_description: str
    example_python_code: str
    high_level_idea: str
    design_justification: str

    @classmethod
    def from_yaml(cls, experiment_dir: str) -> "ExperimentConfig":
        path = Path(experiment_dir) / "config.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        if "CONTEXT" in data:
            context = data["CONTEXT"]
        else:
            context = ""
        return cls(
            context=context,
            function_description=data["FUNCTION_DESCRIPTION"],
            example_python_code=data["EXAMPLE_PYTHON_CODE"],
            high_level_idea=data["HIGH_LEVEL_IDEA"],
            design_justification=data["DESIGN_JUSTIFICATION"],
        )
