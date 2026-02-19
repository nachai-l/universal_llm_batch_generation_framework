from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict

class LLMOutput(BaseModel):
    """
    Represents the structured translation output for interview questions and rubrics.
    All text fields are translated into Thai while maintaining professional tone and specific labels.
    """
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        ..., 
        description="The unique identifier for the question, kept unchanged from input."
    )
    question_name: str = Field(
        ..., 
        description="The translated interview question. Must end with a question mark (?)."
    )
    role_level: str = Field(
        ..., 
        description="The role level description (e.g., Junior, Senior), kept in English as per requirements."
    )
    example_answer_good: str = Field(
        ..., 
        description="The translated high-quality sample answer."
    )
    example_answer_mid: str = Field(
        ..., 
        description="The translated average-quality sample answer."
    )
    example_answer_bad: str = Field(
        ..., 
        description="The translated low-quality sample answer."
    )
    grading_rubrics: str = Field(
        ..., 
        description="The translated grading criteria. Preserves English labels for specific rubric sections."
    )

class JudgeResult(BaseModel):
    """
    Represents the evaluation result from a judge model assessing translation quality.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final judgment on whether the output meets all strict requirements."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="A numerical quality score from 0 to 100."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific justifications or violations found during evaluation."
    )

__all__ = ["LLMOutput", "JudgeResult"]
