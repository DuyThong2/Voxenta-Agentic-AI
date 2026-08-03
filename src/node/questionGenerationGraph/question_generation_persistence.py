from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg

from node.questionGenerationGraph.constants import EMBEDDING_MODEL, TOPICS
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
    TokenCall,
    normalize_name,
    question_embedding_text,
    stable_topic_id,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def index_seed_topics(
    runtime: QuestionGenerationRuntime,
    tokens: list[TokenCall],
) -> None:
    for name, dimension, curriculum_group in TOPICS:
        embedding, token_count = runtime.embed(name)
        tokens.append(
            TokenCall(
                role="embedding",
                mode="topic-upsert",
                model=EMBEDDING_MODEL,
                input=token_count,
                output=0,
                reasoning=0,
                cached_input=0,
                response_id="",
            )
        )
        runtime.topic_collection.upsert(
            ids=[stable_topic_id(name)],
            embeddings=[embedding],
            documents=[name],
            metadatas=[
                {
                    "interest_dimension": dimension,
                    "curriculum_group": curriculum_group,
                    "active": True,
                }
            ],
        )


def index_question(
    runtime: QuestionGenerationRuntime,
    record: dict[str, Any],
    embedding: list[float] | None = None,
) -> TokenCall | None:
    token_call = None
    if embedding is None:
        embedding, token_count = runtime.embed(
            question_embedding_text(
                record["topic_name"],
                record["question_text"],
            )
        )
        token_call = TokenCall(
            role="embedding",
            mode="accepted-upsert",
            model=EMBEDDING_MODEL,
            input=token_count,
            output=0,
            reasoning=0,
            cached_input=0,
            response_id="",
        )
    metadata: dict[str, Any] = {
        "topic_id": record["topic_id"],
        "criterion_code": record["target_criterion_code"],
        "difficulty_rank": record["difficulty_rank"],
        "active": True,
        "usage_count": 0,
    }
    if record["target_sub_attribute"] is not None:
        metadata["sub_attribute"] = record["target_sub_attribute"]
    runtime.question_collection.upsert(
        ids=[record["id"]],
        embeddings=[embedding],
        documents=[
            question_embedding_text(
                record["topic_name"],
                record["question_text"],
            )
        ],
        metadatas=[metadata],
    )
    return token_call


def persist_batch(dsn: str | None, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    if not dsn:
        raise RuntimeError("--persist requires --dsn or VOX_DB_DSN")
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            for record in records:
                cursor.execute(
                    """
                    INSERT INTO new_practice_topic (
                        id, name, normalized_name, description, source,
                        interest_dimension, curriculum_group, active, created_at
                    ) VALUES (%s, %s, %s, %s, 'SEEDED', %s, %s, true, NOW())
                    ON CONFLICT (normalized_name) DO NOTHING
                    """,
                    (
                        record["topic_id"],
                        record["topic_name"],
                        normalize_name(record["topic_name"]),
                        record["topic_name"],
                        record["interest_dimension"],
                        record["curriculum_group"],
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO new_practice_question (
                        id, practice_topic_id, question_text,
                        target_criterion_code, target_sub_attribute,
                        difficulty_rank, difficulty_features_json,
                        evaluation_guide_json, suggested_ideas_json,
                        question_type,
                        min_response_seconds, max_response_seconds,
                        vstep_part, source,
                        usage_count, active, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, 'AI_GENERATED', 0, true, NOW()
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        record["id"],
                        record["topic_id"],
                        record["question_text"],
                        record["target_criterion_code"],
                        record["target_sub_attribute"],
                        record["difficulty_rank"],
                        json.dumps(
                            record["difficulty_features"],
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            record["evaluation_guide"],
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            record["suggested_ideas"],
                            ensure_ascii=False,
                        ),
                        record["question_type"],
                        record["min_response_seconds"],
                        record["max_response_seconds"],
                        record["vstep_part"],
                    ),
                )
