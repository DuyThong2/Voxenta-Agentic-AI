from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriterionCode = Literal[
    "PRONUNCIATION",
    "FLUENCY",
    "GRAMMAR",
    "VOCABULARY",
    "COHERENCE",
]
ReasoningType = Literal[
    "description",
    "comparison",
    "causal",
    "intentional",
    "hypothetical",
]
Abstractness = Literal["concrete_personal", "mixed", "abstract"]


class DifficultyFeatures(BaseModel):
    here_and_now: bool
    num_elements: int = Field(ge=1, le=8)
    reasoning_type: ReasoningType
    abstractness: Abstractness


class EvaluationGuide(BaseModel):
    expected_content: str = Field(min_length=1)
    key_points: str = Field(min_length=1)
    acceptable_responses: str = Field(min_length=1)
    off_topic_examples: str = Field(min_length=1)
    scoring_hints: str = Field(min_length=1)
    common_mistakes: str = Field(min_length=1)


class PracticeQuestionCandidate(BaseModel):
    candidate_id: str
    difficulty_features: DifficultyFeatures
    target_construct: CriterionCode
    target_sub_attribute: str | None = Field(default=None, max_length=64)
    vstep_part: int = Field(ge=1, le=3)
    prompt_text: str = Field(min_length=1)
    suggested_ideas: list[str] = Field(min_length=2, max_length=4)
    planning_time_seconds: int = Field(ge=0, le=120)
    max_response_seconds: int = Field(gt=0, le=300)
    max_followup_seconds: int = Field(ge=0, le=180)
    evaluation_guide: EvaluationGuide


class DraftBatch(BaseModel):
    candidates: list[PracticeQuestionCandidate] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def unique_ids(self) -> "DraftBatch":
        if len({candidate.candidate_id for candidate in self.candidates}) != 3:
            raise ValueError("Drafter candidate IDs must be unique")
        return self


class CandidateVerdict(BaseModel):
    candidate_id: str
    accepted: bool
    violations: list[str]


class EvaluationBatch(BaseModel):
    verdicts: list[CandidateVerdict]


class RefinedBatch(BaseModel):
    candidates: list[PracticeQuestionCandidate] = Field(min_length=1, max_length=4)


def difficulty_rank(features: DifficultyFeatures) -> int:
    reasoning_weight = {
        "description": 0,
        "comparison": 1,
        "causal": 1,
        "intentional": 2,
        "hypothetical": 2,
    }[features.reasoning_type]
    raw = (
        1
        + (not features.here_and_now)
        + (features.num_elements >= 4)
        + reasoning_weight
        + (features.abstractness == "abstract")
    )
    return max(1, min(6, int(raw)))
