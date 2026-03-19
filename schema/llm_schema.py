from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict
from typing import Literal

class LLMOutput(BaseModel):
    """
    Schema for bilingual translation output (English to Thai).
    Ensures structural integrity of translated professional interview questions.
    """
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., description="Unique identifier for the question record.")
    question_set: str = Field(..., description="Name of the source question dataset.")
    question_type: str = Field(..., description="Classification of the question format.")
    assumed_level: str = Field(..., description="Target seniority or difficulty level.")
    question_name: str = Field(..., description="The translated title or content of the question.")
    example_answer_strong: str = Field(..., description="The translated high-quality model answer.")
    example_answer_adequate: str = Field(..., description="The translated satisfactory model answer.")
    example_answer_weak: str = Field(..., description="The translated sub-par model answer.")
    grading_rubrics: str = Field(..., description="The translated evaluation criteria with preserved English headers.")

class JudgeResult(BaseModel):
    """
    Schema for evaluation results produced by a judge model.
    Provides structured feedback and scoring for validation.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(..., description="The final assessment of the generation.")
    score: int = Field(..., ge=0, le=100, description="Quality score from 0 to 100.")
    reasons: List[str] = Field(default_factory=list, description="Detailed explanation of the verdict.")

__all__ = ["LLMOutput", "JudgeResult"]
