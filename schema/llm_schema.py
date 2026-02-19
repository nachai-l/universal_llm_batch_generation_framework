from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict

class LLMOutput(BaseModel):
    """
    Represents the translated interview question content.
    Includes the question, example answers across three quality tiers, and grading rubrics.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ..., 
        description="The unique identifier of the record, preserved from the input."
    )
    question_name: str = Field(
        ..., 
        description="The translated question text in Thai. Must end with a question mark."
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
        description="The translated grading rubrics in Thai, preserving specific English section labels."
    )

class JudgeResult(BaseModel):
    """
    Represents the evaluation result of an LLM-generated translation.
    Used to determine if the output meets the quality and structural requirements.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final decision on whether the output meets all criteria."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="A quality score from 0 to 100 based on translation accuracy and constraint following."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific observations or violations justifying the verdict."
    )

__all__ = ["LLMOutput", "JudgeResult"]
