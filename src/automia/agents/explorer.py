from langchain_core.prompts import ChatPromptTemplate
from ..common import LLMClient, get_logger, GraphState, ExperimentConfig
from pydantic import BaseModel, Field
from ..tools import DatabaseTool

NEW_DESIGN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert researcher designing Membership Inference Attack (MIA) signals. "
     "Your goal is to propose a NOVEL and EFFECTIVE signal that can distinguish member samples from non-member samples "
     "using only the available inputs and context. "
     "Prefer signals that are robust (e.g., to paraphrasing/noise) and not a trivial rewrite of the example."),
    ("user",
    "Design a new MIA signal calculation method according to the spec below.\n\n"
     "Provide the following information. Please be concise and only provide key points:\n"
     "- High-level idea\n"
     "- Design justification (max 300 words)\n"
     "- Implementation instructions\n\n"
     "{experiment_context}"
     "Function specifications:\n"
     "{function_description}\n\n"
     "Reference example (for format/style only; do NOT copy its approach)\n"
     "{example_python_code}\n\n"
     "Previous attempts (do NOT copy):\n"
     "{example_mia_signal_designs}\n\n"
     "Hard constraints for the eventual code implementation:\n"
     "- Python only, executable.\n"
     "- Do NOT include any main/test functions in the code implementation.\n"
     "- Strictly follow the function specifications and input/output specifications. Do NOT deviate from the function specifications and input/output specifications.\n"
     "- Easy to read and easy to modify.\n\n"
     "Additional guidance:\n"
     "- The example code represents the current state of the art in MIA signal design. You MUST propose a better MIA signal design to outperform the SOTA.\n"
     "- You can either propose a new design that inspires from the example code and previous attempts, or propose a completely novel design that is completely different from the example code and previous attempts.\n"
     "- Please refer to the example code for format and input/output specifications.\n"
     "- Keep the signal computationally reasonable for many samples.\n"
     "- Make reasonable default choices of hyperparameters.\n"
     "- Keep implementation instructions brief and focused on the core algorithm.\n"
     "- The proposed signal MUST be different and significant from the previous attempts and likely to outperform the previous attempts.\n"
     "Output a JSON with exactly these fields:\n"
     "- idea: The high-level idea of the MIA signal\n"
     "- design_justification: The design justification of the MIA signal\n"
     "- implementation_instruction: The implementation instruction of the MIA signal\n"
     )
])

REVISE_DESIGN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert researcher designing Membership Inference Attack (MIA) signal designs. "
        "Revise the provided candidate so it becomes meaningfully more novel and effective than prior attempts. "
        "Be concise, keep revisions targeted, and avoid trivial rewrites."
    ),
    (
        "user",
        "Design a new MIA signal calculation method according to the spec below.\n\n"
        "Function specifications:\n"
        "{function_description}\n\n"
        "Reference example (for format/style only; do NOT copy its approach)\n"
        "{example_python_code}\n\n"
        "Relevant previous attempts:\n{relevant_mia_signal_designs}\n\n"
        "Current candidate:\n{current_design}\n\n"
        "Feedback from the nearest neighbor checker:\n{feedback}\n\n"
        "Provide the following information. Please be concise and only provide key points:\n"
        "- High-level idea\n"
        "- Design justification (max 300 words)\n"
        "- Implementation instructions\n\n"
        "{experiment_context}"
        "Hard constraints for the eventual code implementation:\n"
        "- Python only, executable.\n"
        "- Do NOT include any main/test functions in the code implementation.\n"
        "- Strictly follow the function specifications and input/output specifications. Do NOT deviate from the function specifications and input/output specifications.\n"        
        "- Easy to read and easy to modify.\n\n"
        "Additional guidance:\n"
        "- The example code represents the current state of the art in MIA signal design. You MUST propose a better MIA signal design to outperform the SOTA.\n"
        "- Please refer to the example code for format and input/output specifications.\n"
        "- You can either propose a new design that inspires from the example code and previous attempts, or propose a completely novel design that is completely different from the example code and previous attempts.\n"
        "- Keep the signal computationally reasonable for many samples.\n"
        "- Make reasonable default choices of hyperparameters.\n"
        "- Keep implementation instructions brief and focused on the core algorithm.\n"
        "- The proposed signal MUST be different and significant from the previous attempts and likely to outperform the previous attempts.\n"
        "Output a JSON with exactly these fields:\n"
        "- idea: The high-level idea of the MIA signal\n"
        "- design_justification: The design justification of the MIA signal\n"
        "- implementation_instruction: The implementation instruction of the MIA signal\n"
    )
])

class NewDesignOutput(BaseModel):
    idea: str = Field(..., description="The high-level idea of the MIA signal")
    design_justification: str = Field(..., description="The design justification of the MIA signal")
    implementation_instruction: str = Field(..., description="The implementation instruction of the MIA signal")

class NearestNeighborCheckerOutput(BaseModel):
    action: str = Field(..., description="The action to take on the candidate. One of ['accept', 'revise', 'redesign']")
    reasons: str = Field(..., description="The reasoning for the action")
    novelty_score: float = Field(..., description="The novelty score of the candidate with the nearest neighbors")
    suggestions: str = Field(..., description="If action is 'revise', list concrete changes and directions to make it novel. Otherwise, leave this field as an empty string.")

class NewDesignAgent:
    def __init__(self, llm: LLMClient, config: ExperimentConfig, db: DatabaseTool = None):
        self._llm = llm
        self._config = config
        self._logger = get_logger("system")
        if db is not None:
            self._db = db
        else:
            self._db = DatabaseTool()

    def __call__(self):
        self._logger.info("Generating new MIA signal design...")
        experiment_records = self._db.get_random_k_experiments(k=10)
        example_mia_signal_designs = "\n".join([f"Previous attempt {i+1}:\nIdea: {record['idea']}\nDesign justification: {record['design_justification']}\nImplementation: {record['implementation']}, AUC score: {record['auc_score']}, TPR@1 score: {record['tpr_1_score']}, TPR@5 score: {record['tpr_5_score']}, Summary Analysis: {record['analysis_summary']}\n\n" for i, record in enumerate(experiment_records)])

        out = self._llm.invoke_structured(
            NEW_DESIGN_PROMPT,
            {
                "example_mia_signal_designs": example_mia_signal_designs,
                "function_description": self._config.function_description,
                "example_python_code": self._config.example_python_code,
                "experiment_context": self._config.context
            },
            NewDesignOutput,
            "New MIA signal design"
        )

        return out

    def revise_design(self, current_design: NewDesignOutput, relevant_mia_signal_designs: str, feedback: str):
        self._logger.info("Revising MIA signal design...")
        current_design_str = f"Idea: {current_design.idea}\nDesign justification: {current_design.design_justification}\nImplementation instructions: {current_design.implementation_instruction}\n\n"

        out = self._llm.invoke_structured(
            REVISE_DESIGN_PROMPT,
            {
                "function_description": self._config.function_description,
                "example_python_code": self._config.example_python_code,
                "relevant_mia_signal_designs": relevant_mia_signal_designs,
                "current_design": current_design_str,
                "feedback": feedback,
                "experiment_context": self._config.context
            },
            NewDesignOutput,
            "Revise MIA signal design"
        )

        return out

# an agent that will retrieve the nearest neighbors and decide whether to the given MIA signal design is novel or not
NEAREST_NEIGHBOR_CHECKER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict novelty checker for MIA signal designs. Do NOT accept unless similarity is low and the core mechanism is new. "
        "Compare the candidate against prior attempts and decide if it is significantly meaningful different from the prior attempts."
    ),
    (
        "user",
        "Candidate design:\n"
        "- Idea: {idea}\n"
        "- Design justification: {design_justification}\n"
        "- Implementation instructions: {implementation_instruction}\n\n"
        "Nearest prior attempts (most similar first):\n"
        "{relevant_mia_signal_designs}\n\n"
        "Return a JSON with exactly these fields:\n"
        "- action: one of ['accept', 'revise', 'redesign']. `accept` means the candidate is enough novel and should be implemented and run as a new experiment. `revise` means the candidate is not novel enough but can be revised to be novel. `redesign` means the candidate is too similar to the prior attempts, the general approach is not promising, and the entire proposed approach should be redesigned. \n"
        "- reasons: brief reasoning for the action\n"
        "- novelty_score: float in [0,1] where 0 = identical, 1 = unexplored\n"
        "- suggestions: if action is 'revise', list concrete changes and directions to make it novel. Otherwise, leave this field as an empty string."
    ),
])

class NearestNeighborCheckerAgent:
    def __init__(self, llm: LLMClient, db: DatabaseTool):
        self._llm = llm
        self._logger = get_logger("system")
        self._db = db

    def __call__(self, design: NewDesignOutput) -> GraphState:
        self._logger.info("Nearest Neighbor Checker agent: Calling...")
        relevant_idea_records = self._db.get_top_k_nearest_neighbors(query=design.idea, query_type="idea", k=2)
        relevant_design_justifications = self._db.get_top_k_nearest_neighbors(query=design.design_justification, query_type="design_justification", k=2)
        relevant_analysis_summaries = self._db.get_top_k_nearest_neighbors(query=design.design_justification, query_type="analysis_summary", k=2)

        relevant_experiment_records_bm25 = self._db.get_top_k_bm25(query=f"Idea: {design.idea}\nDesign justification: {design.design_justification}", k=5)

        relevant_experiment_records = relevant_idea_records + relevant_design_justifications + relevant_analysis_summaries + relevant_experiment_records_bm25


        unique_ids = set()
        relevant_experiment_records_deduped = []
        for record in relevant_experiment_records:
            if record["id"] not in unique_ids:
                unique_ids.add(record["id"])
                relevant_experiment_records_deduped.append(record)

        relevant_mia_signal_designs = "\n".join([f"Previous attempt {i+1}:\nIdea: {record['idea']}\nDesign justification: {record['design_justification']}\nImplementation: {record['implementation']}, AUC score: {record['auc_score']}, TPR@1 score: {record['tpr_1_score']}, TPR@5 score: {record['tpr_5_score']}, Summary Analysis: {record['analysis_summary']}\n\n" for i, record in enumerate(relevant_experiment_records_deduped)])

        out = self._llm.invoke_structured(
            NEAREST_NEIGHBOR_CHECKER_PROMPT,
            {
                "idea": design.idea,
                "design_justification": design.design_justification,
                "implementation_instruction": design.implementation_instruction,
                "relevant_mia_signal_designs": relevant_mia_signal_designs,
            },
            NearestNeighborCheckerOutput,
            "Nearest Neighbor Checker"
        )

        return out, relevant_mia_signal_designs

class ExplorerAgent:
    def __init__(self, llm: LLMClient, db: DatabaseTool, config: ExperimentConfig, iteration_budget: int = 5):
        self._llm = llm
        self._config = config
        self._logger = get_logger("system")
        self._db = db
        self._new_design_agent = NewDesignAgent(llm, config, db)
        self._nearest_neighbor_checker_agent = NearestNeighborCheckerAgent(llm, db)
        self._iteration_budget = iteration_budget

    def __call__(self, state: GraphState) -> GraphState:
        self._logger.info("\n-------------------------------- Explorer agent START --------------------------------")
        self._logger.info("Explorer agent: Calling...")
        new_design_output = self._new_design_agent()
        for _ in range(self._iteration_budget):
            nearest_neighbor_checker_output, relevant_mia_signal_designs = self._nearest_neighbor_checker_agent(new_design_output)
            if nearest_neighbor_checker_output.action == "accept":
                self._logger.info(f"Novelty score: {nearest_neighbor_checker_output.novelty_score}")
                self._logger.info(f"Reasons: {nearest_neighbor_checker_output.reasons}")
                self._logger.info("Action: Accepting new design")
                break
            elif nearest_neighbor_checker_output.action == "revise":
                self._logger.info(f"Novelty score: {nearest_neighbor_checker_output.novelty_score}")
                self._logger.info(f"Reasons: {nearest_neighbor_checker_output.reasons}")
                self._logger.info(f"Suggestions: {nearest_neighbor_checker_output.suggestions}")
                self._logger.info("Action: Revising design")
                new_design_output = self._new_design_agent.revise_design(new_design_output, relevant_mia_signal_designs, nearest_neighbor_checker_output.suggestions)
            elif nearest_neighbor_checker_output.action == "redesign":
                self._logger.info(f"Novelty score: {nearest_neighbor_checker_output.novelty_score}")
                self._logger.info(f"Reasons: {nearest_neighbor_checker_output.reasons}")
                self._logger.info(f"Suggestions: {nearest_neighbor_checker_output.suggestions}")
                self._logger.info("Action: Redesigning design")
                new_design_output = self._new_design_agent()
        self._logger.info("\n-------------------------------- Explorer agent END --------------------------------\n")

        state.idea = new_design_output.idea
        state.design_justification = new_design_output.design_justification
        state.implementation_instruction = new_design_output.implementation_instruction

        return state
