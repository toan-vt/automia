"""
Automia - Automated Machine Intelligence Agents

A framework for automated machine learning experiments using LLM-powered agents.
"""

__version__ = "0.0.1s"
__author__ = "Automia Team"

# Core components
from .common import (
    register_logger,
    get_logger,
    LLMClient,
    LLMClientProvider,
    GraphState,
    ExperimentConfig,
)

# Agents
from .agents import (
    MutatorAgent,
    CodeGenAgent,
    CodeFixAgent,
    ExecutorAgent,
    ResultReaderAgent,
    ExplorerAgent,
    ExploiterAgent,
)

# Tools
from .tools import (
    DatabaseTool,
    EmbeddingTool,
    ExperimentTool,
    BM25Tool,
)

# Entry point
from .main import main

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Common
    "register_logger",
    "get_logger",
    "LLMClient",
    "LLMClientProvider",
    "GraphState",
    "ExperimentConfig",
    # Agents
    "MutatorAgent",
    "CodeGenAgent",
    "CodeFixAgent",
    "ExecutorAgent",
    "ResultReaderAgent",
    "ExplorerAgent",
    "ExploiterAgent",
    # Tools
    "DatabaseTool",
    "EmbeddingTool",
    "ExperimentTool",
    "BM25Tool",
    # Entry point
    "main",
]
