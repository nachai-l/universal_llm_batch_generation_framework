from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict

class LLMOutput(BaseModel):
    """
    Schema for the translated interview question content.
    Represents the output of the bilingual translation engine.
    """
    model_config = ConfigDict(extra="forbid")

    question_name: str = Field(
        ..., 
        description="The translated interview question name, maintaining interrogative format."
    )
    example_answer_good: str = Field(
        ..., 
        description="The translated high-quality example answer."
    )
    example_answer_mid: str = Field(
        ..., 
        description="The translated medium-quality example answer."
    )
    example_answer_bad: str = Field(
        ..., 
        description="The translated low-quality example answer."
    )
    grading_rubrics: str = Field(
        ..., 
        description="The translated grading rubrics, preserving specific section headers and structure."
    )

class JudgeResult(BaseModel):
    """
    Schema for the judge's evaluation of the LLM output.
    Used to validate the quality and adherence to instructions.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final evaluation outcome."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="A quality score from 0 to 100."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="List of justifications for the verdict and score."
    )

__all__ = ["LLMOutput", "JudgeResult"]
