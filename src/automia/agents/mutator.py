from langchain_core.prompts import ChatPromptTemplate
from ..common import LLMClient, get_logger, GraphState, ExperimentConfig
from pydantic import BaseModel, Field

MUTATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert researcher designing Membership Inference Attack (MIA) signals. "
     "Your goal is to propose a NOVEL and EFFECTIVE signal that can distinguish member samples from non-member samples "
     "using only the available inputs and context. "
     "Prefer signals that are robust (e.g., to paraphrasing/noise) and not a trivial rewrite of the example."),
    ("user",
     "Design a new MIA signal calculation method according to the spec below.\n\n"
     "Provide the following information. Please be concise and only provide key points:\n"
     "- High-level idea\n"
     "- Design justification\n"
     "- Implementation instructions\n\n"
     "Function spec:\n"
     "{function_description}\n\n"
     "Reference example (for format/style only; do NOT copy its approach)\n"
     "{example_python_code}\n\n"
     "Hard constraints for the eventual code implementation:\n"
     "- Python only, executable.\n"
     "- Implement ONLY `get_mia_signal` (no main/test functions).\n"
     "- Easy to read and easy to modify.\n\n"
     "Additional guidance:\n"
     "- Do NOT stick to the example code approach; propose a meaningfully different signal.\n"
     "- Keep the signal computationally reasonable for many samples.\n"
     "- Make reasonable default choices of hyperparameters.\n"
     "- Keep implementation instructions brief and focused on the core algorithm.")
])

class MutatorOutput(BaseModel):
    idea: str = Field(..., description="The high-level idea of the MIA signal")
    design_justification: str = Field(..., description="The design justification of the MIA signal")
    implementation_instruction: str = Field(..., description="The implementation instruction of the MIA signal")

class MutatorAgent:
    def __init__(self, llm: LLMClient, config: ExperimentConfig):
        self._llm = llm
        self._config = config
        self._logger = get_logger("system")

    def __call__(self, state: GraphState) -> GraphState:
        self._logger.info("\n-------------------------------- Mutator agent START --------------------------------")
        self._logger.info("Mutator agent: Calling...")

        out = self._llm.invoke_structured(
            MUTATOR_PROMPT,
            {
                "function_description": self._config.function_description,
                "example_python_code": self._config.example_python_code
            },
            MutatorOutput,
            "MIA signal design"
        )

        state.idea = out.idea
        state.design_justification = out.design_justification
        state.implementation_instruction = out.implementation_instruction

        self._logger.info(f"Mutator agent: MIA signal design completed")
        self._logger.info(f"\t\t idea: {state.idea}")
        self._logger.info(f"\t\t design_justification: {state.design_justification}")
        self._logger.info(f"\t\t implementation_instruction: {state.implementation_instruction}")
        self._logger.info("\n-------------------------------- Mutator agent END --------------------------------\n")
        return state
