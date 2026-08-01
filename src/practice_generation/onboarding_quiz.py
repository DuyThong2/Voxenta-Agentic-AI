"""Generate forced-choice onboarding items offline with verbalized sampling."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

Dimension = Literal[
    "ENTERTAINMENT_MEDIA",
    "TECH_GAMING",
    "SPORTS_HEALTH",
    "PEOPLE_SOCIETY",
    "TRAVEL_PLACES",
    "FUTURE_SCIENCE",
]

SYSTEM_PROMPT = """You write interest-profiler items for Vietnamese high-school students.
Each item is a forced-choice triplet. It measures topic preference, not English ability.
Use concrete, age-appropriate activities. Never copy KOIS wording.

Hard constraints:
- exactly three different dimensions per triplet;
- every statement is Vietnamese and has fewer than 15 whitespace-delimited words;
- all statements have similar social desirability;
- each statement discriminates preference for its assigned dimension;
- no politics, religion, regional stereotypes, family assumptions, economic assumptions,
  specialist knowledge, or required device ownership.

Verbalized sampling:
For every requested group, produce five substantially different candidate triplets and
assign probabilities that sum to 1. Do not explain chain-of-thought. The caller samples
one candidate from your stated distribution."""

USER_PROMPT = """Generate {count} candidate groups.

Examples of the desired activity style, not wording to repeat:
- "Xem hậu trường một bộ phim mới"
- "Thử chiến thuật cho một môn thể thao"
- "Tìm hiểu cách một phát minh hoạt động"

Make groups diverse across all six dimensions. Avoid repeating the same activity or
sentence across groups."""


class DesirabilityCheck(BaseModel):
    balanced: bool
    note: str = Field(min_length=1, max_length=240)


class TripletCandidate(BaseModel):
    probability: float = Field(gt=0, le=1)
    dimension_per_statement: list[Dimension] = Field(min_length=3, max_length=3)
    statements: list[str] = Field(min_length=3, max_length=3)
    desirability_check: DesirabilityCheck

    @model_validator(mode="after")
    def validate_triplet(self) -> "TripletCandidate":
        if len(set(self.dimension_per_statement)) != 3:
            raise ValueError("Every triplet must contain three distinct dimensions")
        if not self.desirability_check.balanced:
            raise ValueError("Social desirability must be balanced")
        for statement in self.statements:
            if len(statement.split()) >= 15:
                raise ValueError(f"Statement has 15 words or more: {statement}")
        return self


class CandidateGroup(BaseModel):
    candidates: list[TripletCandidate] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "CandidateGroup":
        total = sum(candidate.probability for candidate in self.candidates)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Candidate probabilities sum to {total}, not 1")
        return self


class QuizGenerationBatch(BaseModel):
    groups: list[CandidateGroup]


def select_items(
    batch: QuizGenerationBatch,
    *,
    count: int,
    seed: int,
) -> list[TripletCandidate]:
    if len(batch.groups) != count:
        raise ValueError(f"Expected {count} groups, received {len(batch.groups)}")
    generator = random.Random(seed)
    selected = [
        generator.choices(
            group.candidates,
            weights=[candidate.probability for candidate in group.candidates],
            k=1,
        )[0]
        for group in batch.groups
    ]
    normalized_triplets = {
        tuple(statement.casefold().strip() for statement in item.statements)
        for item in selected
    }
    if len(normalized_triplets) != len(selected):
        raise ValueError("Selected triplets contain duplicates")
    normalized_statements = [
        statement.casefold().strip()
        for item in selected
        for statement in item.statements
    ]
    if len(set(normalized_statements)) != len(normalized_statements):
        raise ValueError("Selected triplets reuse a statement")
    return selected


def generate(
    *,
    count: int = 20,
    seed: int = 20260729,
    model: str = "gpt-5.4",
) -> tuple[QuizGenerationBatch, list[TripletCandidate], dict[str, int]]:
    client = OpenAI()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(count=count)},
        ],
        text_format=QuizGenerationBatch,
    )
    if response.output_parsed is None:
        raise RuntimeError("Quiz generator returned no structured output")
    selected = select_items(response.output_parsed, count=count, seed=seed)
    usage = response.usage
    tokens = {
        "input": usage.input_tokens if usage else 0,
        "output": usage.output_tokens if usage else 0,
        "reasoning": (
            usage.output_tokens_details.reasoning_tokens
            if usage and usage.output_tokens_details
            else 0
        ),
    }
    return response.output_parsed, selected, tokens


def write_output(
    output_dir: Path,
    batch: QuizGenerationBatch,
    selected: list[TripletCandidate],
    tokens: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates.json").write_text(
        batch.model_dump_json(indent=2),
        encoding="utf-8",
    )
    selected_payload = [
        {
            "dimension_per_statement": item.dimension_per_statement,
            "statements": item.statements,
            "desirability_check": item.desirability_check.model_dump(),
        }
        for item in selected
    ]
    (output_dir / "selected.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tokens.json").write_text(
        json.dumps(tokens, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seed-output")
        / datetime.now(UTC).date().isoformat()
        / "interest-quiz",
    )
    args = parser.parse_args()
    batch, selected, tokens = generate(
        count=args.count,
        seed=args.seed,
        model=args.model,
    )
    write_output(args.output, batch, selected, tokens)
    print(
        json.dumps(
            {
                "selected": len(selected),
                "unique_triplets": len(
                    {tuple(item.statements) for item in selected}
                ),
                "tokens": tokens,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
