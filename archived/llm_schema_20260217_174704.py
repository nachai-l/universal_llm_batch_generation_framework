from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict

class QuestionItem(BaseModel):
    """Represents a single generated interview question and its evaluation metadata."""
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        ..., 
        description="The unique identifier for the question, preserved from the input context."
    )
    question_name: str = Field(
        ..., 
        description="A short, descriptive title for the question topic."
    )
    question_text: str = Field(
        ..., 
        description="The actual interrogative interview question. Must be at least 10 words and end with a question mark."
    )
    example_answer_good: str = Field(
        ..., 
        description="A high-quality example answer demonstrating desired competency."
    )
    example_answer_mid: str = Field(
        ..., 
        description="An average example answer showing basic understanding but lacking depth."
    )
    example_answer_bad: str = Field(
        ..., 
        description="A poor example answer or one containing red flags."
    )
    grading_rubrics: str = Field(
        ..., 
        description="Detailed grading criteria with 'Good:', 'To avoid:', and 'Red Flag:' sections."
    )

class LLMOutput(BaseModel):
    """The root schema for the LLM's structured question generation response."""
    model_config = ConfigDict(extra="forbid")

    questions: List[QuestionItem] = Field(
        ..., 
        description="A list containing all generated questions for the specified role."
    )

class JudgeResult(BaseModel):
    """The schema for the evaluation output of the judge model."""
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final pass/fail decision for the generated content."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="The numerical quality score from 0 to 100."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific reasons or feedback supporting the verdict."
    )

__all__ = ["LLMOutput", "JudgeResult", "QuestionItem"]
