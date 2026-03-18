from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict
from typing import Literal

class LLMOutput(BaseModel):
    """
    Schema for the translated interview question content from English to Thai.
    Includes the question, example answers of varying quality, and grading rubrics.
    """
    model_config = ConfigDict(extra="forbid")

    question_name: str = Field(
        ..., 
        description="The translated interview question name. Must end with a question mark."
    )
    example_answer_good: str = Field(
        ..., 
        description="The translated professional, high-quality example answer."
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
        description="The translated grading rubrics including required labels for intent, structure, depth, clarity, and relevance."
    )

class JudgeResult(BaseModel):
    """
    Schema for the evaluation result provided by the judge model.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final pass/fail decision based on the translation quality."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="The quality score from 0 to 100."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific reasons or feedback for the verdict."
    )

__all__ = ["LLMOutput", "JudgeResult"]
