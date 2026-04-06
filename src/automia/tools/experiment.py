import os
import shlex
import subprocess
from typing import Optional
from ..common import get_logger
from pathlib import Path

class ExperimentTool:
    def __init__(self, timeout: int, bash_env_script_path: str = None):
        """
        Args:
            timeout: The timeout for the experiment in seconds.
            bash_script_path: The path to the bash script to run the environment setup.
        """
        self._logger = get_logger("system")
        self._timeout = timeout
        self._bash_env_script_path = bash_env_script_path
        # check if the env script path exists
        if self._bash_env_script_path and not os.path.exists(self._bash_env_script_path):
            self._logger.warning(f"Environment setup script not found: {self._bash_env_script_path}")
        
    def __call__(self, runtime_dir: str) -> str:
        # mia_run.py should receive the output directory as an argument
        # it will save the results in the output-dir/mia-results.json
        log_file_path = Path(runtime_dir) / "experiment.log"
        try:
            self._logger.info(f"Running environment setup script: {self._bash_env_script_path}")
            self._logger.info(f"Experiment logging to: {log_file_path}")
            # Run env setup script then the experiment python in one bash session
            if self._bash_env_script_path:
                cmd = (
                        f"source {shlex.quote(self._bash_env_script_path)}"
                        f" && python {shlex.quote(runtime_dir)}/mia_run.py --output-dir {shlex.quote(runtime_dir)}"
                    )
            else:
                cmd = (
                    f"python {shlex.quote(runtime_dir)}/mia_run.py --output-dir {shlex.quote(runtime_dir)}"
                )
            with open(log_file_path, 'w') as log_file:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    check=False,
                    cwd=".",
                    timeout=self._timeout
                )
                
        except subprocess.TimeoutExpired:
            self._logger.error(f"TIMEOUT: Experiment timed out after {self._timeout} seconds. This approach is too slow and needs a new idea.")
            return "TIMEOUT: Experiment timed out. This approach is too slow and needs a new idea."
        except Exception as e:
            self._logger.error(f"Error running the experiment: {e}")
            return f"Error running the experiment: {e}"

        if result.returncode != 0:
            # the log file of the last 30 lines
            with open(log_file_path, 'r') as f:
                log_lines = f.readlines()[-30:]
                log_lines = "\n".join(log_lines)
                num_lines = len(log_lines.split("\n"))
                error_message = f"Error running the experiment, Experiment failed with return code {result.returncode}. Check log file at {log_file_path}.\nLast {num_lines} lines of the log file:\n{log_lines}"
                self._logger.error(error_message)
                return error_message

        output_path = Path(runtime_dir)/"mia-results.json"
        if not output_path.exists():
            self._logger.error(f"Error running the experiment, output file not found. Searched for: {output_path}")
            return f"Error running the experiment, output file not found. Searched for: {output_path}"
        with open(output_path, 'r') as f:
            return f.read()
