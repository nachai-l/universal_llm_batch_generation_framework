from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict

class LLMOutput(BaseModel):
    """
    Schema for the generated interview question and example answers.
    Used for strict validation of LLM generation output.
    """
    model_config = ConfigDict(extra="forbid")

    question_name: str = Field(
        ..., 
        description="A clear, realistic interview question aligned with the role and level."
    )
    example_answer_good: str = Field(
        ..., 
        description="A high-quality (7-10) structured and relevant answer for a junior candidate."
    )
    example_answer_mid: str = Field(
        ..., 
        description="A medium-quality (4-6) generic answer that lacks specificity or depth."
    )
    example_answer_bad: str = Field(
        ..., 
        description="A low-quality (3 or less) vague or off-topic answer."
    )
    grading_rubrics: str = Field(
        ..., 
        description="Concise rubric sections including Good, To avoid, and Red Flag criteria."
    )

class JudgeResult(BaseModel):
    """
    Schema for the judge LLM output when evaluating responses.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final pass or fail judgment based on evaluation criteria."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="A numerical score between 0 and 100."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific reasons or justifications for the score and verdict."
    )

__all__ = ["LLMOutput", "JudgeResult"]
