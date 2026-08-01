from __future__ import annotations

import math
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from openai import OpenAI

from node.questionGenerationGraph.constants import (
    EMBEDDING_MODEL,
    MODEL,
)
from schemas.question_generation import (
    CandidateVerdict,
    PracticeQuestionCandidate,
    difficulty_rank,
)
from vector.chroma_client import build_raw_collection


@dataclass
class TokenCall:
    role: str
    mode: str
    model: str
    input: int
    output: int
    reasoning: int
    cached_input: int
    response_id: str


class QuestionGenerationRuntime:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.question_collection = build_raw_collection(
            "practice_questions",
            embedding_model=EMBEDDING_MODEL,
        )
        self.topic_collection = build_raw_collection(
            "practice_topics",
            embedding_model=EMBEDDING_MODEL,
        )

    def parsed_call(
        self,
        *,
        role: str,
        mode: str,
        effort: str,
        system: str,
        prompt: str,
        schema: Any,
        tokens: list[TokenCall],
    ) -> tuple[Any, dict[str, Any]]:
        started = time.perf_counter()
        response = self.client.responses.parse(
            model=MODEL,
            reasoning={"effort": effort},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            text_format=schema,
        )
        if response.output_parsed is None:
            raise RuntimeError(f"{role} returned no structured output")
        usage = response.usage
        input_details = usage.input_tokens_details if usage else None
        output_details = usage.output_tokens_details if usage else None
        call = TokenCall(
            role=role,
            mode=mode,
            model=MODEL,
            input=usage.input_tokens if usage else 0,
            output=usage.output_tokens if usage else 0,
            reasoning=output_details.reasoning_tokens if output_details else 0,
            cached_input=input_details.cached_tokens if input_details else 0,
            response_id=response.id,
        )
        tokens.append(call)
        return response.output_parsed, {
            "response_id": response.id,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "parsed": response.output_parsed.model_dump(),
            "tokens": asdict(call),
        }

    def embed(self, text: str) -> tuple[list[float], int]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding, response.usage.prompt_tokens

    def max_similarity(self, embedding: list[float]) -> float:
        if self.question_collection.count() == 0:
            return 0.0
        result = self.question_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"active": True},
            include=["distances"],
        )
        distances = result.get("distances") or []
        if not distances or not distances[0]:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(distances[0][0])))


def question_embedding_text(topic_name: str, prompt_text: str) -> str:
    return f"Topic: {topic_name}\nQuestion: {prompt_text}"


def stable_topic_id(name: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"vox-practice-topic:{normalize_name(name)}",
        )
    )


def normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD",
        value.casefold().strip().replace("đ", "d"),
    )
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_marks.split())


def verdict_signature(
    verdict: CandidateVerdict,
) -> tuple[bool, tuple[str, ...]]:
    return verdict.accepted, tuple(sorted(verdict.violations))


def question_record(
    *,
    question_id: str,
    topic: tuple[str, str, str],
    candidate: PracticeQuestionCandidate,
    topic_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": question_id,
        "candidate_id": candidate.candidate_id,
        "topic_id": topic_id or stable_topic_id(topic[0]),
        "topic_name": topic[0],
        "interest_dimension": topic[1],
        "curriculum_group": topic[2],
        "question_text": candidate.prompt_text,
        "target_criterion_code": candidate.target_construct,
        "target_sub_attribute": candidate.target_sub_attribute,
        "difficulty_rank": difficulty_rank(candidate.difficulty_features),
        "difficulty_features": candidate.difficulty_features.model_dump(),
        "evaluation_guide": candidate.evaluation_guide.model_dump(),
        "suggested_ideas": candidate.suggested_ideas,
        "preparation_time_seconds": candidate.planning_time_seconds,
        "max_response_seconds": candidate.max_response_seconds,
        "max_followup_seconds": candidate.max_followup_seconds,
        "vstep_part": candidate.vstep_part,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def rank_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {str(rank): 0 for rank in range(1, 7)}
    for record in records:
        result[str(record["difficulty_rank"])] += 1
    return result
