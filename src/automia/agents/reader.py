from langchain_core.prompts import ChatPromptTemplate
from ..common import LLMClient, get_logger, GraphState, ExperimentConfig
from pydantic import BaseModel, Field

RESULT_READER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert researcher specializing in membership inference attacks (MIA) and machine learning security. "
     "Your role is to critically analyze MIA experiment results and provide comprehensive insights that will inform future research directions."),
    ("user",
     "Analyze the following MIA experiment and provide a structured summary of the findings.\n\n"
     "MIA Design Information:\n"
     "- Design Idea: {idea}\n"
     "- Design Justification: {design_justification}\n"
     "- Implementation Code: {code_block}\n"
     "- Experiment Results: {results}\n\n"
     "{experiment_context}"
     "Your analysis should be concise and focus into the key points (max 300 words):\n"
     "1. Evaluate the effectiveness of the MIA signal\n"
     "2. Identify key insights about what makes this signal work or fail\n"
     "3. Highlight limitations and potential failure modes\n"
     "4. Note any novel or innovative aspects of the approach\n\n"),
])

class ResultReaderOutput(BaseModel):
    analysis_summary: str = Field(..., description="A concise overview of the analysis highlighting key insights, limitations, and innovations.")

class ResultReaderAgent:
    def __init__(self, llm: LLMClient, config: ExperimentConfig):
        self._llm = llm
        self._config = config
        self._logger = get_logger("system")

    def __call__(self, state: GraphState) -> GraphState:
        self._logger.info("\n-------------------------------- Result Reader agent START --------------------------------")
        self._logger.info("Result Reader agent: Calling...")
        out = self._llm.invoke_structured(
            RESULT_READER_PROMPT,
            {
                "idea": state.idea,
                "design_justification": state.design_justification,
                "code_block": state.code_block,
                "results": state.experiment_result,
                "experiment_context": self._config.context
            },
            ResultReaderOutput,
            "Result Reader"
        )
        state.analysis_summary = out.analysis_summary
        self._logger.info("Result Reader agent: Completed.")
        self._logger.info(f"\t\t analysis_summary: {state.analysis_summary}")
        self._logger.info("\n-------------------------------- Result Reader agent END --------------------------------\n")
        return state