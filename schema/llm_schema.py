from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict

class QuestionItem(BaseModel):
    """Represents a single interview question and its associated evaluation data."""
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        ..., 
        description="The unique identifier for the question."
    )
    question_name: str = Field(
        ..., 
        description="The full interrogative sentence of the interview question."
    )
    example_answer_good: str = Field(
        ..., 
        description="A high-quality example response that meets all criteria."
    )
    example_answer_mid: str = Field(
        ..., 
        description="A mediocre example response that is acceptable but lacking depth."
    )
    example_answer_bad: str = Field(
        ..., 
        description="A poor example response that demonstrates lack of skill or red flags."
    )
    grading_rubrics: str = Field(
        ..., 
        description="Structured rubrics containing 'Good:', 'To avoid:', and 'Red Flag:' sections."
    )

class LLMOutput(BaseModel):
    """The structured output containing the full set of generated interview questions."""
    model_config = ConfigDict(extra="forbid")

    questions: List[QuestionItem] = Field(
        ..., 
        description="A list of generated interview questions for the specified role."
    )

class JudgeResult(BaseModel):
    """The result of an LLM-based evaluation or grading process."""
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final pass or fail judgment."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="A numerical score between 0 and 100."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific reasons or feedback points supporting the verdict."
    )

__all__ = ["LLMOutput", "JudgeResult"]
