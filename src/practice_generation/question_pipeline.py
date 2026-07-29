"""Offline MAFIG-style practice-question generation with auditable backups."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from openai import OpenAI

from config.chroma_config import settings
from practice_generation.models import (
    CandidateVerdict,
    DraftBatch,
    EvaluationBatch,
    PracticeQuestionCandidate,
    RefinedBatch,
    difficulty_rank,
)
from vector.chroma_client import build_raw_collection

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = os.getenv("PRACTICE_GENERATION_MODEL", "gpt-5.4")
EMBEDDING_MODEL = settings.OPENAI_EMBEDDING_MODEL
HARD_CAP = 50
DRAFTER_CANDIDATES = 3
MAX_EDITOR_ROUNDS = 3
REFINER_BATCH_SIZE = 4
DUPLICATE_THRESHOLD = 0.92

BAND_LADDER = """SIX-BAND SPEAKING LADDER - KEEP THIS PREFIX EXACT
1 BAC_1: concrete, immediate personal information; short simple descriptions.
2 BAC_2: familiar matters; connected basic details with limited reasons.
3 BAC_3: familiar and some less familiar matters; compare options and explain reasons.
4 BAC_4: develop a clear argument; handle abstraction and causal relationships.
5 BAC_5: flexible, precise discussion of complex or hypothetical implications.
6 BAC_6: nuanced synthesis, subtle distinctions, and well-controlled complex reasoning.
END SIX-BAND SPEAKING LADDER"""

SAFETY_CONSTRAINTS = """Reject any question that assumes overseas travel, family structure,
economic resources, device ownership, specialist knowledge, politics, religion, or regional
stereotypes. The prompt must be neutral and answerable by every Vietnamese high-school student."""

TOPICS = [
    ("School clubs", "PEOPLE_SOCIETY", "IN_GDPT2018"),
    ("Healthy routines", "SPORTS_HEALTH", "IN_GDPT2018"),
    ("Films and stories", "ENTERTAINMENT_MEDIA", "OUT_OF_CURRICULUM"),
    ("Games and technology", "TECH_GAMING", "OUT_OF_CURRICULUM"),
    ("Places in my town", "TRAVEL_PLACES", "IN_GDPT2018"),
    ("Everyday science", "FUTURE_SCIENCE", "IN_GDPT2018"),
    ("Learning with friends", "PEOPLE_SOCIETY", "IN_GDPT2018"),
    ("Sports choices", "SPORTS_HEALTH", "IN_GDPT2018"),
    ("Music and performance", "ENTERTAINMENT_MEDIA", "OUT_OF_CURRICULUM"),
    ("Future inventions", "FUTURE_SCIENCE", "IN_GDPT2018"),
]

CRITERIA = [
    ("PRONUNCIATION", None),
    ("FLUENCY", None),
    ("GRAMMAR", "sv_agreement"),
    ("GRAMMAR", "tense_control"),
    ("GRAMMAR", "complex_clause_control"),
    ("GRAMMAR", "third_person_s_omission"),
    ("GRAMMAR", "article_use"),
    ("GRAMMAR", "word_form"),
    ("VOCABULARY", "limited_range"),
    ("VOCABULARY", "repetition"),
    ("VOCABULARY", "weak_collocation"),
    ("COHERENCE", "weak_progression"),
    ("COHERENCE", "limited_support"),
    ("COHERENCE", "connector_overuse"),
    ("COHERENCE", "topic_drift"),
]

ALLOWED_SUB_ATTRIBUTES: dict[str, frozenset[str | None]] = {
    "PRONUNCIATION": frozenset({None}),
    "FLUENCY": frozenset({None}),
    "GRAMMAR": frozenset(
        {
            "sv_agreement",
            "tense_control",
            "complex_clause_control",
            "third_person_s_omission",
            "article_use",
            "word_form",
        }
    ),
    "VOCABULARY": frozenset(
        {"limited_range", "repetition", "weak_collocation"}
    ),
    "COHERENCE": frozenset(
        {
            "weak_progression",
            "limited_support",
            "connector_overuse",
            "topic_drift",
        }
    ),
}
FILTER_REASON_CODES = frozenset(
    {
        "NOT_ENGLISH",
        "LENGTH_OUT_OF_RANGE",
        "MISSING_FIELD",
        "SUB_ATTRIBUTE_NOT_ALLOWED",
        "CRITERION_UNKNOWN",
        "DUPLICATE_COSINE",
    }
)


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


class GenerationPipeline:
    def __init__(
        self,
        *,
        output_root: Path,
        persist: bool,
        dsn: str | None,
    ) -> None:
        self.client = OpenAI()
        self.output_root = output_root
        self.persist = persist
        self.dsn = dsn
        self.question_collection = build_raw_collection(
            "practice_questions",
            embedding_model=EMBEDDING_MODEL,
        )
        self.topic_collection = build_raw_collection(
            "practice_topics",
            embedding_model=EMBEDDING_MODEL,
        )
        self.accepted_count = 0
        self.batch_number = 0
        self.all_tokens: list[TokenCall] = []
        self.cosines: list[float] = []
        self.editor_rounds: list[int] = []
        self.candidate_total = 0
        self.evaluator_rejected = 0
        self.comparison_total = 0
        self.comparison_different = 0
        self.accepted_questions: list[dict[str, Any]] = []
        self.zero_batch_reasons: list[set[str]] = []

    def run(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._resume_existing_batches()
        self._index_topics()
        combination_index = 0
        while self.accepted_count < HARD_CAP:
            topic = TOPICS[combination_index % len(TOPICS)]
            criterion = CRITERIA[combination_index % len(CRITERIA)]
            target_rank = 1 + combination_index % 6
            combination_index += 1
            self.batch_number += 1
            accepted_in_batch, filter_reasons = self._run_batch(
                topic,
                criterion,
                target_rank,
            )
            self._check_systematic_filter_failure(
                accepted_in_batch,
                filter_reasons,
            )
            if self.batch_number >= 100 and self.accepted_count < HARD_CAP:
                raise RuntimeError("Hard safety stop: 100 batches without 50 accepted questions")
        summary = self._summary()
        self._write_json(self.output_root / "summary.json", summary)
        return summary

    def _resume_existing_batches(self) -> None:
        for batch_dir in sorted(self.output_root.glob("batch-*")):
            required = [
                batch_dir / "drafter-raw.json",
                batch_dir / "evaluator-raw.json",
                batch_dir / "editor-raw.json",
                batch_dir / "accepted.json",
                batch_dir / "tokens.json",
            ]
            if not all(path.exists() for path in required):
                continue
            try:
                number = int(batch_dir.name.removeprefix("batch-"))
            except ValueError:
                continue
            self.batch_number = max(self.batch_number, number)
            draft = json.loads(required[0].read_text(encoding="utf-8"))
            parsed_draft = draft.get("parsed", draft)
            self.candidate_total += len(parsed_draft.get("candidates", []))

            evaluator = json.loads(required[1].read_text(encoding="utf-8"))
            separate_signatures: dict[str, tuple[bool, tuple[str, ...]]] = {}
            for call in evaluator.get("separate", []):
                for verdict in call.get("parsed", {}).get("verdicts", []):
                    signature = (
                        bool(verdict.get("accepted")),
                        tuple(sorted(verdict.get("violations", []))),
                    )
                    separate_signatures[verdict["candidate_id"]] = signature
                    if not verdict.get("accepted"):
                        self.evaluator_rejected += 1
            grouped_call = evaluator.get("grouped") or {}
            for verdict in grouped_call.get("parsed", {}).get("verdicts", []):
                separate = separate_signatures.get(verdict["candidate_id"])
                if separate is None:
                    continue
                grouped = (
                    bool(verdict.get("accepted")),
                    tuple(sorted(verdict.get("violations", []))),
                )
                self.comparison_total += 1
                self.comparison_different += separate != grouped
            self.cosines.extend(
                float(value)
                for value in evaluator.get("vector_similarity", {}).values()
            )

            editor = json.loads(required[2].read_text(encoding="utf-8"))
            rounds_by_candidate: dict[str, int] = defaultdict(int)
            for item in editor.get("calls", []):
                if "round" not in item:
                    continue
                candidate_id = (
                    item.get("editor", {})
                    .get("parsed", {})
                    .get("candidate_id")
                )
                if candidate_id:
                    rounds_by_candidate[candidate_id] = max(
                        rounds_by_candidate[candidate_id],
                        int(item["round"]),
                    )
            self.editor_rounds.extend(
                rounds_by_candidate.get(candidate_id, 0)
                for candidate_id in separate_signatures
            )

            accepted = json.loads(required[3].read_text(encoding="utf-8"))
            self.accepted_questions.extend(accepted)
            self.accepted_count += len(accepted)

            token_payload = json.loads(required[4].read_text(encoding="utf-8"))
            for call in token_payload.get("calls", []):
                self.all_tokens.append(TokenCall(**call))

    def _run_batch(
        self,
        topic: tuple[str, str, str],
        criterion: tuple[str, str | None],
        target_rank: int,
    ) -> tuple[int, set[str]]:
        batch_dir = self.output_root / f"batch-{self.batch_number:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        tokens: list[TokenCall] = []
        draft, draft_raw = self._draft(topic, criterion, target_rank, tokens)
        self.candidate_total += len(draft.candidates)
        self._write_json(batch_dir / "drafter-raw.json", draft_raw)

        survivors: list[PracticeQuestionCandidate] = []
        rejected: list[dict[str, Any]] = []
        filter_reasons: set[str] = set()
        survivor_embeddings: dict[str, list[float]] = {}
        for candidate in draft.candidates:
            violations = rule_violations(candidate)
            if violations:
                reason, detail = violations[0]
                filter_reasons.update(item[0] for item in violations)
                rejected.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "reason": reason,
                        "detail": detail,
                        "candidate": candidate.model_dump(),
                    }
                )
                continue
            embedding, embed_tokens = self._embed(
                question_embedding_text(topic[0], candidate.prompt_text)
            )
            tokens.append(
                TokenCall(
                    role="embedding",
                    mode="question-filter",
                    model=EMBEDDING_MODEL,
                    input=embed_tokens,
                    output=0,
                    reasoning=0,
                    cached_input=0,
                    response_id="",
                )
            )
            similarity = self._max_similarity(embedding)
            self.cosines.append(similarity)
            if similarity >= DUPLICATE_THRESHOLD:
                filter_reasons.add("DUPLICATE_COSINE")
                rejected.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "reason": "DUPLICATE_COSINE",
                        "detail": (
                            f"cosine {similarity:.6f} >= "
                            f"{DUPLICATE_THRESHOLD:.2f}"
                        ),
                        "candidate": candidate.model_dump(),
                    }
                )
                continue
            survivors.append(candidate)
            survivor_embeddings[candidate.candidate_id] = embedding

        separate_verdicts: dict[str, CandidateVerdict] = {}
        separate_raw: list[dict[str, Any]] = []
        for candidate in survivors:
            evaluation, raw = self._evaluate(
                [candidate],
                topic,
                target_rank,
                mode="separate",
                tokens=tokens,
            )
            verdict = evaluation.verdicts[0]
            separate_verdicts[verdict.candidate_id] = verdict
            separate_raw.append(raw)

        grouped_raw: dict[str, Any] | None = None
        grouped_verdicts: dict[str, CandidateVerdict] = {}
        if survivors:
            grouped, grouped_raw = self._evaluate(
                survivors,
                topic,
                target_rank,
                mode="grouped",
                tokens=tokens,
            )
            grouped_verdicts = {
                verdict.candidate_id: verdict for verdict in grouped.verdicts
            }
            for candidate in survivors:
                separate = separate_verdicts[candidate.candidate_id]
                grouped_item = grouped_verdicts.get(candidate.candidate_id)
                self.comparison_total += 1
                if grouped_item is None or verdict_signature(separate) != verdict_signature(
                    grouped_item
                ):
                    self.comparison_different += 1

        evaluator_payload: dict[str, Any] = {
            "separate": separate_raw,
            "grouped": grouped_raw,
            "rejected_before_evaluator": rejected,
            "vector_similarity": {
                candidate_id: self._max_similarity(embedding)
                for candidate_id, embedding in survivor_embeddings.items()
            },
        }
        self._write_json(batch_dir / "evaluator-raw.json", evaluator_payload)

        edited_raw: list[dict[str, Any]] = []
        live: list[PracticeQuestionCandidate] = []
        for candidate in survivors:
            verdict = separate_verdicts[candidate.candidate_id]
            rounds = 0
            current = candidate
            while not verdict.accepted and rounds < MAX_EDITOR_ROUNDS:
                self.evaluator_rejected += 1
                rounds += 1
                current, editor_raw = self._edit(
                    current,
                    verdict,
                    topic,
                    target_rank,
                    rounds,
                    tokens,
                )
                evaluation, raw = self._evaluate(
                    [current],
                    topic,
                    target_rank,
                    mode=f"post-editor-{rounds}",
                    tokens=tokens,
                )
                verdict = evaluation.verdicts[0]
                edited_raw.append(
                    {
                        "round": rounds,
                        "editor": editor_raw,
                        "evaluator": raw,
                    }
                )
            self.editor_rounds.append(rounds)
            if verdict.accepted:
                live.append(current)
            else:
                rejected.append(
                    {
                        "candidate": current.model_dump(),
                        "stage": "editor-limit",
                        "violations": verdict.violations,
                    }
                )

        refined: list[PracticeQuestionCandidate] = []
        for start in range(0, len(live), REFINER_BATCH_SIZE):
            chunk = live[start : start + REFINER_BATCH_SIZE]
            polished, raw = self._refine(chunk, topic, target_rank, tokens)
            refined.extend(polished.candidates)
            edited_raw.append({"refiner": raw})

        accepted_payload: list[dict[str, Any]] = []
        for candidate in refined:
            violations = rule_violations(candidate)
            if violations:
                reason, detail = violations[0]
                filter_reasons.update(item[0] for item in violations)
                rejected.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "reason": reason,
                        "detail": f"post-refiner validation: {detail}",
                        "candidate": candidate.model_dump(),
                    }
                )
                continue
            if self.accepted_count >= HARD_CAP:
                rejected.append(
                    {
                        "candidate": candidate.model_dump(),
                        "stage": "hard-cap",
                        "violations": ["hard_cap_50_reached"],
                    }
                )
                break
            question_id = str(uuid.uuid4())
            record = question_record(
                question_id=question_id,
                topic=topic,
                candidate=candidate,
            )
            accepted_payload.append(record)
            self.accepted_questions.append(record)
            self.accepted_count += 1

        self._write_json(
            batch_dir / "editor-raw.json",
            {"calls": edited_raw, "rejected": rejected},
        )
        self._write_json(batch_dir / "accepted.json", accepted_payload)
        self._write_json(
            batch_dir / "tokens.json",
            {"calls": [asdict(call) for call in tokens]},
        )

        # The five backup files above are durable before either DB or Chroma changes.
        if self.persist:
            self._persist_batch(accepted_payload)
        for record in accepted_payload:
            candidate_id = record["candidate_id"]
            embedding = survivor_embeddings.get(candidate_id)
            if embedding is None:
                embedding, embed_tokens = self._embed(
                    question_embedding_text(
                        record["topic_name"],
                        record["question_text"],
                    )
                )
                tokens.append(
                    TokenCall(
                        role="embedding",
                        mode="accepted-upsert",
                        model=EMBEDDING_MODEL,
                        input=embed_tokens,
                        output=0,
                        reasoning=0,
                        cached_input=0,
                        response_id="",
                    )
                )
                self._write_json(
                    batch_dir / "tokens.json",
                    {"calls": [asdict(call) for call in tokens]},
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
            self.question_collection.upsert(
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
        self.all_tokens.extend(tokens)
        return len(accepted_payload), filter_reasons

    def _check_systematic_filter_failure(
        self,
        accepted_in_batch: int,
        filter_reasons: set[str],
    ) -> None:
        if accepted_in_batch:
            self.zero_batch_reasons.clear()
            return
        self.zero_batch_reasons.append(filter_reasons)
        self.zero_batch_reasons = self.zero_batch_reasons[-3:]
        if len(self.zero_batch_reasons) < 3:
            return
        shared = set.intersection(*self.zero_batch_reasons)
        if shared:
            reason = sorted(shared)[0]
            raise RuntimeError(
                "Stopped after three consecutive zero-output batches "
                f"sharing filter reason {reason}"
            )

    def _draft(
        self,
        topic: tuple[str, str, str],
        criterion: tuple[str, str | None],
        target_rank: int,
        tokens: list[TokenCall],
    ) -> tuple[DraftBatch, dict[str, Any]]:
        prompt = f"""Generate exactly {DRAFTER_CANDIDATES} different English speaking questions.
Topic: {topic[0]}
Target construct: {criterion[0]}
Target sub-attribute: {json.dumps(criterion[1])}
Target cognitive rank: approximately {target_rank}; fill difficulty_features honestly.
{SAFETY_CONSTRAINTS}

Use verbalized sampling internally: consider varied approaches, then return three candidates.
Do not return difficulty_rank. Do not return followup_questions.
Each evaluation guide must have all six non-empty fields.
The target_sub_attribute taxonomy is closed:
- PRONUNCIATION and FLUENCY: null only.
- GRAMMAR: sv_agreement, tense_control, complex_clause_control,
  third_person_s_omission, article_use, word_form.
- VOCABULARY: limited_range, repetition, weak_collocation.
- COHERENCE: weak_progression, limited_support, connector_overuse, topic_drift.
Return exactly the target sub-attribute shown above; never invent another value."""
        return self._parsed_call(
            role="drafter",
            mode="batch",
            effort="low",
            system=(
                "You draft inclusive English speaking-practice questions. "
                "Return structured data only."
            ),
            prompt=prompt,
            schema=DraftBatch,
            tokens=tokens,
        )

    def _evaluate(
        self,
        candidates: list[PracticeQuestionCandidate],
        topic: tuple[str, str, str],
        target_rank: int,
        *,
        mode: str,
        tokens: list[TokenCall],
    ) -> tuple[EvaluationBatch, dict[str, Any]]:
        prompt = f"""{BAND_LADDER}

EVALUATION RULES
Return one verdict per candidate. accepted=true only when violations is empty.
Check topic fit, target construct, target sub-attribute, neutral access, internal consistency,
all six evaluation-guide fields, and whether difficulty_features imply rank near {target_rank}.
The target_sub_attribute taxonomy is closed. PRONUNCIATION and FLUENCY require null;
null is correct for those two constructs and must never be reported as missing. GRAMMAR allows
sv_agreement, tense_control, complex_clause_control, third_person_s_omission, article_use,
word_form. VOCABULARY allows limited_range, repetition, weak_collocation. COHERENCE allows
weak_progression, limited_support, connector_overuse, topic_drift.
{SAFETY_CONSTRAINTS}
Do not assign a total quality score. Return concrete violation codes.

TOPIC
{topic[0]}

CANDIDATES TO EVALUATE
{json.dumps([item.model_dump() for item in candidates], ensure_ascii=False)}"""
        return self._parsed_call(
            role="evaluator",
            mode=mode,
            effort="high",
            system=(
                "You are a strict independent evaluator. "
                "The six-band ladder at the beginning of the user message is authoritative."
            ),
            prompt=prompt,
            schema=EvaluationBatch,
            tokens=tokens,
        )

    def _edit(
        self,
        candidate: PracticeQuestionCandidate,
        verdict: CandidateVerdict,
        topic: tuple[str, str, str],
        target_rank: int,
        round_number: int,
        tokens: list[TokenCall],
    ) -> tuple[PracticeQuestionCandidate, dict[str, Any]]:
        prompt = f"""Edit this candidate to remove every listed violation.
Keep candidate_id unchanged. Do not return difficulty_rank or followup_questions.
Topic: {topic[0]}. Target rank: {target_rank}. Editor round: {round_number}.
Violations: {json.dumps(verdict.violations, ensure_ascii=False)}
Candidate: {candidate.model_dump_json()}
PRONUNCIATION and FLUENCY require target_sub_attribute=null. Preserve that null value.
{SAFETY_CONSTRAINTS}"""
        return self._parsed_call(
            role="editor",
            mode=f"round-{round_number}",
            effort="low",
            system="You repair a question without changing its intended construct.",
            prompt=prompt,
            schema=PracticeQuestionCandidate,
            tokens=tokens,
        )

    def _refine(
        self,
        candidates: list[PracticeQuestionCandidate],
        topic: tuple[str, str, str],
        target_rank: int,
        tokens: list[TokenCall],
    ) -> tuple[RefinedBatch, dict[str, Any]]:
        prompt = f"""Polish these {len(candidates)} independent accepted questions in one pass.
Preserve candidate IDs, constructs, sub-attributes, difficulty features, time budgets, and
evaluation-guide meaning. Improve only clarity and natural English. Do not add
followup_questions or difficulty_rank. Topic: {topic[0]}. Target rank: {target_rank}.
Candidates: {json.dumps([item.model_dump() for item in candidates], ensure_ascii=False)}"""
        return self._parsed_call(
            role="refiner",
            mode=f"batch-{len(candidates)}",
            effort="low",
            system="You perform light independent copy-editing on accepted questions.",
            prompt=prompt,
            schema=RefinedBatch,
            tokens=tokens,
        )

    def _parsed_call(
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

    def _embed(self, text: str) -> tuple[list[float], int]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding, response.usage.prompt_tokens

    def _max_similarity(self, embedding: list[float]) -> float:
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

    def _index_topics(self) -> None:
        for name, dimension, curriculum_group in TOPICS:
            topic_id = stable_topic_id(name)
            embedding, tokens = self._embed(name)
            self.all_tokens.append(
                TokenCall(
                    role="embedding",
                    mode="topic-upsert",
                    model=EMBEDDING_MODEL,
                    input=tokens,
                    output=0,
                    reasoning=0,
                    cached_input=0,
                    response_id="",
                )
            )
            self.topic_collection.upsert(
                ids=[topic_id],
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

    def _persist_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        if not self.dsn:
            raise RuntimeError("--persist requires --dsn or VOX_DB_DSN")
        with psycopg.connect(self.dsn) as connection:
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
                            preparation_time_seconds, max_response_seconds,
                            max_followup_seconds, vstep_part, source,
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
                            record["preparation_time_seconds"],
                            record["max_response_seconds"],
                            record["max_followup_seconds"],
                            record["vstep_part"],
                        ),
                    )

    def _summary(self) -> dict[str, Any]:
        calls = [asdict(call) for call in self.all_tokens]
        totals = {
            "input": sum(call.input for call in self.all_tokens),
            "output": sum(call.output for call in self.all_tokens),
            "reasoning": sum(call.reasoning for call in self.all_tokens),
            "cached_input": sum(call.cached_input for call in self.all_tokens),
        }
        role_totals: dict[str, dict[str, int]] = {}
        for call in self.all_tokens:
            aggregate = role_totals.setdefault(
                call.role,
                {"input": 0, "output": 0, "reasoning": 0, "cached_input": 0},
            )
            aggregate["input"] += call.input
            aggregate["output"] += call.output
            aggregate["reasoning"] += call.reasoning
            aggregate["cached_input"] += call.cached_input
        return {
            "hard_cap": HARD_CAP,
            "accepted": self.accepted_count,
            "candidate_total": self.candidate_total,
            "success_ratio_p": (
                self.accepted_count / self.candidate_total
                if self.candidate_total
                else 0.0
            ),
            "evaluator_rejection_ratio": (
                self.evaluator_rejected / self.candidate_total
                if self.candidate_total
                else 0.0
            ),
            "average_editor_rounds": (
                statistics.mean(self.editor_rounds)
                if self.editor_rounds
                else 0.0
            ),
            "evaluator_comparison": {
                "total": self.comparison_total,
                "different": self.comparison_different,
                "difference_ratio": (
                    self.comparison_different / self.comparison_total
                    if self.comparison_total
                    else 0.0
                ),
            },
            "cosine": {
                "count": len(self.cosines),
                "p95": percentile(self.cosines, 0.95),
                "max": max(self.cosines, default=0.0),
            },
            "tokens": {
                "totals": totals,
                "by_role": role_totals,
                "calls": calls,
            },
            "difficulty_rank_distribution": rank_distribution(
                self.accepted_questions
            ),
            "persisted": self.persist,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": MODEL,
        }

    def _write_json(self, path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def rule_violations(
    candidate: PracticeQuestionCandidate,
) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    text = candidate.prompt_text.strip()
    words = text.split()
    ascii_letters = sum(character.isascii() and character.isalpha() for character in text)
    letters = sum(character.isalpha() for character in text)
    if len(words) < 6 or len(words) > 80:
        violations.append(
            (
                "LENGTH_OUT_OF_RANGE",
                f"prompt has {len(words)} words; expected 6..80",
            )
        )
    if letters == 0 or ascii_letters / letters < 0.9:
        violations.append(
            (
                "NOT_ENGLISH",
                "fewer than 90% of alphabetic characters are ASCII English",
            )
        )
    allowed = ALLOWED_SUB_ATTRIBUTES.get(candidate.target_construct)
    if allowed is None:
        violations.append(
            (
                "CRITERION_UNKNOWN",
                f"{candidate.target_construct} is not a framework criterion",
            )
        )
    elif candidate.target_sub_attribute not in allowed:
        rendered = (
            "null"
            if candidate.target_sub_attribute is None
            else candidate.target_sub_attribute
        )
        violations.append(
            (
                "SUB_ATTRIBUTE_NOT_ALLOWED",
                f"{rendered} is not allowed for {candidate.target_construct}",
            )
        )
    assert all(reason in FILTER_REASON_CODES for reason, _ in violations)
    return violations


def question_embedding_text(topic_name: str, prompt_text: str) -> str:
    return f"Topic: {topic_name}\nQuestion: {prompt_text}"


def stable_topic_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vox-practice-topic:{normalize_name(name)}"))


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


def verdict_signature(verdict: CandidateVerdict) -> tuple[bool, tuple[str, ...]]:
    return verdict.accepted, tuple(sorted(verdict.violations))


def question_record(
    *,
    question_id: str,
    topic: tuple[str, str, str],
    candidate: PracticeQuestionCandidate,
) -> dict[str, Any]:
    return {
        "id": question_id,
        "candidate_id": candidate.candidate_id,
        "topic_id": stable_topic_id(topic[0]),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seed-output") / date.today().isoformat(),
    )
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--dsn", default=os.getenv("VOX_DB_DSN"))
    args = parser.parse_args()
    pipeline = GenerationPipeline(
        output_root=args.output,
        persist=args.persist,
        dsn=args.dsn,
    )
    summary = pipeline.run()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
