from pydantic import BaseModel, Field


class GraphState(BaseModel):
    idea: str = Field(..., description="The high-level idea of the MIA signal")
    design_justification: str = Field(..., description="The design justification of the MIA signal")
    implementation_instruction: str = Field(..., description="The implementation instruction of the signal")
    code_block: str = Field(..., description="The complete Python code block with imports")
    error_flag: bool = Field(..., description="Whether the code block has an error")
    error_message: str = Field(..., description="The error message of the code block")
    fix_attempts: int = Field(..., description="The number of fix attempts")
    experiment_result: dict = Field(..., description="The result of the experiment")
    analysis_summary: str = Field(..., description="The analysis summary of the experiment")
    experiment_id: int = Field(..., description="The id of the experiment")
