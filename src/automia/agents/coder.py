from langchain_core.prompts import ChatPromptTemplate
from ..common import LLMClient, get_logger, GraphState, ExperimentConfig
from pydantic import BaseModel, Field

CODE_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software engineer specializing in Python. "
     "Your job is to implement a Python function that computes a membership inference (MIA) signal score "
     "for a prompt-completion setting. "
     "Write clean, readable, and easily modifiable code that is fully executable."),

    ("user",
     "Implement the MIA signal function according to the spec and instructions below.\n\n"
     "## Function spec\n"
     "{function_description}\n\n"
     "## Reference example (for format/style only; do NOT copy its approach)\n"
     "{example_python_code}\n\n"
     "## High-level idea of the MIA signal\n"
     "{idea}\n\n"
     "## Implementation requirements\n"
     "{implementation_instruction}\n\n"
     "## Code requirements\n"
     "- Return ONLY valid Python code block as plain text (no Markdown fences, no extra prose).\n"
     "- Put all required imports at the top.\n"
     "- Provide concise, high-level comments only where helpful. Avoid excessive or line-by-line comments.\n"
     "- Implement the function exactly as specified (name/signature/return type).\n"
     "- Do NOT include any main/test functions in the code implementation.\n"
     "- Do NOT perform I/O (no printing, files, network) and do NOT rely on global state.\n"
     "- Never return None or an empty response; always return a finite float for any input\n"
     "- Do NOT use try-except blocks to handle errors and exceptions.\n"
     "- The environment is fixed. If the code block imports a library that is not installed, modify the code to not use that library.\n"
     "- Thinking carefully about efficiency. If a pretrained model is used, consider declaring it as a global variable to avoid re-loading it multiple times.\n"
     "## Output requirements\n"
     "- A single Python code block with the required function and all required imports at the top.\n")
])

class CodeGenOutput(BaseModel):
    code_block: str = Field(..., description="The complete Python code block with imports")

class CodeGenAgent:
    def __init__(self, llm: LLMClient, config: ExperimentConfig):
        self._llm = llm
        self._config = config
        self._logger = get_logger("system")

    def __call__(self, state: GraphState) -> GraphState:
        self._logger.info("\n-------------------------------- Coder agent START --------------------------------")
        self._logger.info("Coder agent: Calling...")

        out = self._llm.invoke_structured(
            CODE_GEN_PROMPT,
            {
                "function_description": self._config.function_description,
                "example_python_code": self._config.example_python_code,
                "idea": state.idea,
                "implementation_instruction": state.implementation_instruction,
            },
            CodeGenOutput,
            "Code generation"
        )

        state.code_block = out.code_block
        self._logger.info("Coder agent: Code generation completed")
        self._logger.info(f"\t\t code_block: {state.code_block}")
        self._logger.info("\n-------------------------------- Coder agent END --------------------------------\n")

        return state

CODE_FIX_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software engineer specializing in Python. "
     "Your job is to fix the code block that is provided to you. "
     "Write clean, readable, and easily modifiable code that is fully executable."),

    ("user",
     "Final goal: Implement a *new* MIA signal function according to the spec below.\n\n"
     "## Function spec\n"
     "{function_description}\n\n"
     "## Reference example (for format/style only; do NOT copy its approach)\n"
     "{example_python_code}\n\n"
     "## High-level idea of the MIA signal\n"
     "{idea}\n\n"
     "## Implementation requirements\n"
     "{implementation_instruction}\n\n"
     "## Current buggy code block\n"
     "{code_block}\n\n"
     "## Error message\n"
     "{error_message}\n\n"
     "## Guidance\n"
     "- Given the error message, you should fix the code block to be executable and correct. Please follow the original idea and implementation instructions as much as possible.\n"
     "- If timeout, consider changing the hyperparameters to reduce the computation time.\n"
     "## Code requirements\n"
     "- Strictly follow the function specifications and input/output specifications. Do NOT deviate from the function specifications and input/output specifications.\n"
     "- Put all required imports at the top.\n"
     "- Provide concise, high-level comments only where helpful. Avoid excessive or line-by-line comments.\n"
     "- Implement the function exactly as specified (name/signature/return type).\n"
     "- Implement ONLY the required function (no main/test functions).\n"
     "- Do NOT perform I/O (no printing, files, network) and do NOT rely on global state.\n"
     "- Never return None or an empty response; always return a finite float for any input\n"
     "- Do NOT use try-except blocks to handle errors and exceptions.\n"
     "- The environment is fixed. If the code block imports a library that is not installed, modify the code to not use that library.\n"
     "- Thinking carefully about efficiency. If a pretrained model is used, consider declaring it as a global variable to avoid re-loading it multiple times.\n"
     "## Output requirements, a JSON with the following fields\n"
     "1. error_diagnosis: A clear error summary\n"
     "2. changes_made: The changes made to the code block\n"
     "3. code_block: The complete fixed Python code block with the required function and all required imports at the top\n"
    )
])

class CodeFixOutput(BaseModel):
    error_diagnosis: str = Field(..., description="The error diagnosis of the code block")
    changes_made: str = Field(..., description="The changes made to the code block")
    code_block: str = Field(..., description="The complete Python code block with imports")

class CodeFixAgent:
    def __init__(self, llm: LLMClient, config: ExperimentConfig):
        self._llm = llm
        self._config = config
        self._logger = get_logger("system")

    def __call__(self, state: GraphState) -> GraphState:
        self._logger.info("\n-------------------------------- Code Fix agent START --------------------------------")
        self._logger.info(f"Code Fix agent: Fixing code block (attempt {state.fix_attempts + 1})")
        out = self._llm.invoke_structured(
            CODE_FIX_PROMPT,
            {
                "function_description": self._config.function_description,
                "example_python_code": self._config.example_python_code,
                "idea": state.idea,
                "implementation_instruction": state.implementation_instruction,
                "code_block": state.code_block,
                "error_message": state.error_message,
            },
            CodeFixOutput,
            "Code fixing"
        )

        state.fix_attempts += 1
        state.code_block = out.code_block

        self._logger.info(f"Code Fix agent: Error diagnosis: {out.error_diagnosis}")
        self._logger.info(f"Code Fix agent: Changes made: {out.changes_made}")
        self._logger.info(f"Code Fix agent: Fixed code block:\n{state.code_block}")
        self._logger.info("\n-------------------------------- Code Fix agent END --------------------------------\n")
        return state
