import logging
import sys
from pathlib import Path
from typing import Optional, Dict

_FMT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_LOG_FILES: Dict[str, str] = {}

def register_logger(name: str, file_path: str) -> None:
    """Call once (e.g., in main.py) to bind a logger name -> file."""
    _LOG_FILES[name] = str(Path(file_path).resolve())

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Call anywhere. No need to pass file_path again."""
    if name not in _LOG_FILES:
        raise ValueError(
            f"Logger '{name}' is not registered. "
            f"Call register_logger('{name}', <file_path>) at startup."
        )

    file_path = _LOG_FILES[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent propagation to root logger
    needs_stdout = name == "system"

    # Keep only handlers we care about (matching file + optional stdout)
    handlers_to_keep = []
    has_file = False
    has_stdout = False
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == file_path:
            has_file = True
            handlers_to_keep.append(h)
        elif needs_stdout and isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
            has_stdout = True
            handlers_to_keep.append(h)

    logger.handlers.clear()
    for h in handlers_to_keep:
        logger.addHandler(h)

    if not has_file:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(_FMT)
        logger.addHandler(fh)

    if needs_stdout and not has_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(_FMT)
        logger.addHandler(sh)

    return logger
