from ..common import get_logger, GraphState
from ..tools import ExperimentTool, DatabaseTool
import os
import time
from pathlib import Path
import json
import uuid


def read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

class ExecutorAgent:
    def __init__(self, experiment_template_file_path: str, output_dir: str, timeout: int = 5*60, bash_env_script_path: str = None, db: DatabaseTool = None):
        self._experiment_template_file_path = experiment_template_file_path
        self._output_dir = output_dir
        self._logger = get_logger("system")
        self._experiment_tool = ExperimentTool(timeout=timeout, bash_env_script_path=bash_env_script_path)
        if db is not None:
            self._db = db
        else:
            self._db = DatabaseTool()

    def __call__(self, state: GraphState) -> GraphState:
        self._logger.info("\n-------------------------------- Executor agent START --------------------------------")
        self._logger.info(f"Executor agent: Running experiment ....")
        unique_suffix = uuid.uuid4().hex[:8]
        pid_time_dir = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}_{unique_suffix}"
        self._runtime_output_dir = Path(self._output_dir) / "runtime" / pid_time_dir
        self._runtime_output_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_output_dir = str(self._runtime_output_dir)
        self._runtime_file_path =  f"{self._runtime_output_dir}/mia_run.py"

        template_code = read_text(self._experiment_template_file_path)
        if state.code_block.startswith("```python"):
            state.code_block = state.code_block[len("```python"):]
        if state.code_block.endswith("```"):
            state.code_block = state.code_block[:-len("```")]
        if state.code_block.startswith("python"):
            state.code_block = state.code_block[len("python"):]
        updated_code = template_code.replace("### <CODE-BLOCK>", state.code_block)

        write_text(self._runtime_file_path, updated_code)
        exp_log = self._experiment_tool(self._runtime_output_dir)

        if exp_log.startswith("Error running the experiment"):
            state.error_flag = True
            state.error_message = exp_log
            self._logger.info(f"Executor agent: Experiment failed with error: {exp_log}")
        elif exp_log.startswith("TIMEOUT:"):
            state.error_flag = True
            state.error_message = exp_log + "\n<TIMEOUT>"
            self._logger.info(f"Executor agent: Experiment failed with TIMEOUT")
        else:
            state.error_flag = False
            state.error_message = ""
            state.fix_attempts = 0
            state.experiment_result = json.loads(exp_log)
            self._logger.info(f"Executor agent: Experiment completed successfully. The results of this MIA signal design:\n{state.experiment_result}")
        
        self._logger.info("\n-------------------------------- Executor agent END --------------------------------\n")

        return state