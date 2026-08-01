from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from node.questionGenerationGraph.CandidateFilterNode.candidate_filter_node_config import (
    rule_violations,
)
from node.questionGenerationGraph.constants import (
    CRITERIA,
    EMBEDDING_MODEL,
    HARD_CAP,
    TOPICS,
)
from node.questionGenerationGraph.graphConfig import QuestionGenerationGraph
from node.questionGenerationGraph.question_generation_backup import (
    resume_existing_batches,
)
from node.questionGenerationGraph.question_generation_graph_helper import (
    TokenCall,
    question_record,
)
from node.questionGenerationGraph.question_generation_metrics import (
    build_summary,
)
from node.questionGenerationGraph.question_generation_persistence import (
    index_question,
    index_seed_topics,
    persist_batch,
    write_json,
)


class GenerationPipeline:
    def __init__(
        self,
        *,
        output_root: Path,
        persist: bool,
        dsn: str | None,
    ) -> None:
        self.graph = QuestionGenerationGraph()
        self.output_root = output_root
        self.persist = persist
        self.dsn = dsn
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
        resume_existing_batches(self, self.output_root)
        index_seed_topics(self.graph.runtime, self.all_tokens)
        combination_index = self.accepted_count
        while self.accepted_count < HARD_CAP:
            topic = TOPICS[combination_index % len(TOPICS)]
            criterion = CRITERIA[combination_index % len(CRITERIA)]
            target_rank = 3 + combination_index % 3
            combination_index += 1
            self.batch_number += 1
            accepted, reasons = self._run_batch(
                topic,
                criterion,
                target_rank,
            )
            self._check_systematic_filter_failure(accepted, reasons)
            if self.batch_number >= 125 and self.accepted_count < HARD_CAP:
                raise RuntimeError(
                    "Hard safety stop: 125 batches without 80 accepted questions"
                )
        summary = build_summary(self)
        write_json(self.output_root / "summary.json", summary)
        return summary

    def _run_batch(
        self,
        topic: tuple[str, str, str],
        criterion: tuple[str, str | None],
        target_rank: int,
    ) -> tuple[int, set[str]]:
        batch_dir = self.output_root / f"batch-{self.batch_number:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        state = self.graph.invoke(topic, criterion, target_rank)
        self.candidate_total += len(state["candidates"])
        self.cosines.extend(state["cosines"])
        self.editor_rounds.extend(state["editor_rounds"])
        self.evaluator_rejected += state["evaluator_rejected"]
        self.comparison_total += state["comparison_total"]
        self.comparison_different += state["comparison_different"]

        write_json(batch_dir / "drafter-raw.json", state["drafter_raw"])
        write_json(batch_dir / "evaluator-raw.json", state["evaluator_raw"])
        accepted = self._accepted_records(state, topic)
        write_json(
            batch_dir / "editor-raw.json",
            {"calls": state["editor_raw"], "rejected": state["rejected"]},
        )
        write_json(batch_dir / "accepted.json", accepted)
        write_json(
            batch_dir / "tokens.json",
            {"calls": [asdict(call) for call in state["token_calls"]]},
        )

        if self.persist:
            persist_batch(self.dsn, accepted)
        for record in accepted:
            embedding = state["survivor_embeddings"].get(
                record["candidate_id"]
            )
            token_call = index_question(
                self.graph.runtime,
                record,
                embedding,
            )
            if token_call is not None:
                state["token_calls"].append(token_call)
        write_json(
            batch_dir / "tokens.json",
            {"calls": [asdict(call) for call in state["token_calls"]]},
        )
        self.all_tokens.extend(state["token_calls"])
        return len(accepted), state["filter_reasons"]

    def _accepted_records(self, state: dict, topic: tuple) -> list[dict]:
        records = []
        for candidate in state["refined"]:
            violations = rule_violations(candidate)
            if violations:
                state["filter_reasons"].update(
                    reason for reason, _ in violations
                )
                state["rejected"].append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "reason": violations[0][0],
                        "detail": (
                            "post-refiner validation: "
                            + violations[0][1]
                        ),
                        "candidate": candidate.model_dump(),
                    }
                )
                continue
            if self.accepted_count >= HARD_CAP:
                state["rejected"].append(
                    {
                        "candidate": candidate.model_dump(),
                        "stage": "hard-cap",
                        "violations": ["hard_cap_80_reached"],
                    }
                )
                break
            record = question_record(
                question_id=str(uuid.uuid4()),
                topic=topic,
                candidate=candidate,
            )
            records.append(record)
            self.accepted_questions.append(record)
            self.accepted_count += 1
        return records

    def _check_systematic_filter_failure(
        self,
        accepted: int,
        reasons: set[str],
    ) -> None:
        if accepted:
            self.zero_batch_reasons.clear()
            return
        self.zero_batch_reasons = [
            *self.zero_batch_reasons[-2:],
            reasons,
        ]
        if len(self.zero_batch_reasons) < 3:
            return
        shared = set.intersection(*self.zero_batch_reasons)
        if shared:
            raise RuntimeError(
                "Stopped after three consecutive zero-output batches "
                f"sharing filter reason {sorted(shared)[0]}"
            )
