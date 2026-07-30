import json
from collections import defaultdict
from pathlib import Path

from node.questionGenerationGraph.question_generation_graph_helper import (
    TokenCall,
)


def resume_existing_batches(pipeline, output_root: Path) -> None:
    for batch_dir in sorted(output_root.glob("batch-*")):
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
        pipeline.batch_number = max(pipeline.batch_number, number)
        _resume_drafter(pipeline, required[0])
        signatures = _resume_evaluator(pipeline, required[1])
        _resume_editor(pipeline, required[2], signatures)
        _resume_accepted(pipeline, required[3])
        _resume_tokens(pipeline, required[4])


def _resume_drafter(pipeline, path: Path) -> None:
    draft = json.loads(path.read_text(encoding="utf-8"))
    parsed = draft.get("parsed", draft)
    pipeline.candidate_total += len(parsed.get("candidates", []))


def _resume_evaluator(pipeline, path: Path) -> dict:
    evaluator = json.loads(path.read_text(encoding="utf-8"))
    separate_signatures = {}
    for call in evaluator.get("separate", []):
        for verdict in call.get("parsed", {}).get("verdicts", []):
            signature = (
                bool(verdict.get("accepted")),
                tuple(sorted(verdict.get("violations", []))),
            )
            separate_signatures[verdict["candidate_id"]] = signature
            if not verdict.get("accepted"):
                pipeline.evaluator_rejected += 1
    grouped_call = evaluator.get("grouped") or {}
    for verdict in grouped_call.get("parsed", {}).get("verdicts", []):
        separate = separate_signatures.get(verdict["candidate_id"])
        if separate is None:
            continue
        grouped = (
            bool(verdict.get("accepted")),
            tuple(sorted(verdict.get("violations", []))),
        )
        pipeline.comparison_total += 1
        pipeline.comparison_different += separate != grouped
    pipeline.cosines.extend(
        float(value)
        for value in evaluator.get("vector_similarity", {}).values()
    )
    return separate_signatures


def _resume_editor(pipeline, path: Path, signatures: dict) -> None:
    editor = json.loads(path.read_text(encoding="utf-8"))
    rounds_by_candidate: dict[str, int] = defaultdict(int)
    for item in editor.get("calls", []):
        if "round" not in item:
            continue
        candidate_id = (
            item.get("editor", {}).get("parsed", {}).get("candidate_id")
        )
        if candidate_id:
            rounds_by_candidate[candidate_id] = max(
                rounds_by_candidate[candidate_id],
                int(item["round"]),
            )
    pipeline.editor_rounds.extend(
        rounds_by_candidate.get(candidate_id, 0)
        for candidate_id in signatures
    )


def _resume_accepted(pipeline, path: Path) -> None:
    accepted = json.loads(path.read_text(encoding="utf-8"))
    pipeline.accepted_questions.extend(accepted)
    pipeline.accepted_count += len(accepted)


def _resume_tokens(pipeline, path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for call in payload.get("calls", []):
        pipeline.all_tokens.append(TokenCall(**call))
