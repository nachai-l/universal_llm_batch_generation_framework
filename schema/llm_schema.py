from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict
from typing import Literal

class LLMOutput(BaseModel):
    """
    Schema for the translated interview question generation.
    Enforces bilingual translation rules and structure preservation.
    """
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        ..., 
        description="The unique identifier for the question, must remain unchanged."
    )
    question_name: str = Field(
        ..., 
        description="The translated interview question name in Thai. Must end with a question mark."
    )
    example_answer_good: str = Field(
        ..., 
        description="The translated high-quality example answer in Thai."
    )
    example_answer_mid: str = Field(
        ..., 
        description="The translated average-quality example answer in Thai."
    )
    example_answer_bad: str = Field(
        ..., 
        description="The translated low-quality example answer in Thai."
    )
    grading_rubrics: str = Field(
        ..., 
        description="The translated grading rubrics in Thai, preserving English section labels."
    )

class JudgeResult(BaseModel):
    """
    Schema for the validation and grading of the LLM translation output.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="Binary verdict based on translation quality and rule adherence."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Numerical score representing translation accuracy and naturalness."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="List of specific reasons or violations found during judging."
    )

__all__ = ["LLMOutput", "JudgeResult"]
