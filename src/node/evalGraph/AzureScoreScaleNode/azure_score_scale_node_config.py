"""Convert Azure's HundredMark scores to each rubric criterion's score range."""

from typing import Any, Dict, Optional

from node.evalGraph.PronunciationNode.pronunciation_node_helper import (
    build_framework_note,
)
from schemas.framework import CriterionFramework


def scale_hundred_mark_score(
    azure_score: Optional[float],
    score_min: float,
    score_max: float,
) -> Optional[float]:
    """Map a bounded Azure 0-100 score linearly into [score_min, score_max]."""
    if azure_score is None:
        return None
    if score_max <= score_min:
        raise ValueError("rubric criterion max score must be greater than min score")

    bounded_azure_score = min(100.0, max(0.0, float(azure_score)))
    scaled = score_min + (bounded_azure_score / 100.0) * (score_max - score_min)
    return round(min(score_max, max(score_min, scaled)), 2)


def _find_framework(
    criteria_frameworks: list[CriterionFramework],
    criterion_key: str,
) -> Optional[CriterionFramework]:
    return next(
        (
            framework
            for framework in criteria_frameworks
            if framework.criterion_key == criterion_key
        ),
        None,
    )


def _scale_criterion(
    pronunciation_result: Any,
    criteria_frameworks: list[CriterionFramework],
    criterion_key: str,
) -> Optional[Dict[str, float]]:
    framework = _find_framework(criteria_frameworks, criterion_key)
    if framework is None:
        return None

    criterion = getattr(pronunciation_result.criteria, criterion_key)
    raw_score = criterion.score
    scaled_score = scale_hundred_mark_score(
        raw_score,
        framework.rubric_min_score,
        framework.rubric_max_score,
    )
    if scaled_score is None:
        return None

    criterion.score = scaled_score
    criterion.source = "azure"
    criterion.status = "scored"
    criterion.subscores = {
        **(criterion.subscores or {}),
        "raw_azure_score": raw_score,
    }
    framework_note = build_framework_note(
        criteria_frameworks,
        criterion_key,
        scaled_score,
    )
    if framework_note:
        criterion.note = framework_note

    return {
        "rawAzureScore": raw_score,
        "rubricScore": scaled_score,
        "rubricMinScore": framework.rubric_min_score,
        "rubricMaxScore": framework.rubric_max_score,
    }


def azure_score_scale_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Scale Azure-owned pronunciation and fluency criteria before score merge."""
    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")
    if speaking_input is None or pronunciation_result is None:
        return {
            "metadata": {
                "azure_score_scale_error": (
                    "speaking_input and pronunciation_result are required"
                )
            }
        }

    try:
        scaling_details: Dict[str, Dict[str, float]] = {}
        for criterion_key in ("pronunciation", "fluency"):
            details = _scale_criterion(
                pronunciation_result,
                speaking_input.criteria_frameworks or [],
                criterion_key,
            )
            if details is not None:
                scaling_details[criterion_key] = details

        return {
            "pronunciation_result": pronunciation_result,
            "metadata": {"azure_score_scaling": scaling_details},
        }
    except (TypeError, ValueError) as exc:
        return {"metadata": {"azure_score_scale_error": str(exc)}}
