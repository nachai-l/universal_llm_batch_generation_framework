from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import ConfigDict

class QuestionItem(BaseModel):
    """
    Represents an individual interview question with metadata and grading criteria.
    """
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        ..., 
        description="The unique identifier for the question, preserved from context."
    )
    question_name: str = Field(
        ..., 
        description="The interrogative interview question text (at least 10 words, ends with ?)."
    )
    example_answer_good: str = Field(
        ..., 
        description="A strong, concrete, and structured response following the STAR method."
    )
    example_answer_mid: str = Field(
        ..., 
        description="A mediocre response that is partially structured but lacks depth or specific details."
    )
    example_answer_bad: str = Field(
        ..., 
        description="A poor response that is vague, generic, or misaligned with the role expectations."
    )
    grading_rubrics: str = Field(
        ..., 
        description="Detailed evaluation criteria covering intent, structure, depth, clarity, and relevance."
    )

class LLMOutput(BaseModel):
    """
    The complete structured output containing a list of generated interview questions.
    """
    model_config = ConfigDict(extra="forbid")

    questions: List[QuestionItem] = Field(
        ..., 
        description="A list of question objects generated for the specified role."
    )

class JudgeResult(BaseModel):
    """
    The structured output for evaluating or judging a specific input or response.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL"] = Field(
        ..., 
        description="The final pass/fail decision based on the evaluation."
    )
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="The numerical score assigned to the output (0 to 100)."
    )
    reasons: List[str] = Field(
        default_factory=list, 
        description="A list of specific justifications for the assigned score and verdict."
    )

__all__ = ["LLMOutput", "JudgeResult", "QuestionItem"]
