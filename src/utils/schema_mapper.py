"""Shared schema mapping helpers for assessment validity and scoring payloads."""

from typing import Any, Dict, List, Optional

from schemas.validity import RuleResult, ValidityResult

VALID_ACTIONS = {"score", "score_with_penalty", "teacher_review", "reject_or_zero"}
VALID_SEVERITIES = {"none", "low", "medium", "high", "critical"}
VALID_CRITERIA = {"pronunciation", "fluency", "grammar", "vocabulary", "coherence"}


def safe_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def safe_text(value: Any) -> str:
    return str(value) if value is not None else ""


def normalize_rule_action(action: Any) -> str:
    if isinstance(action, str) and action in VALID_ACTIONS:
        return action
    return "reject_or_zero"


def normalize_rule_severity(severity: Any) -> str:
    if isinstance(severity, str) and severity in VALID_SEVERITIES:
        return severity
    return "critical"


def normalize_target_criteria(targets: Any) -> List[str]:
    if not isinstance(targets, list):
        return []
    return [str(item) for item in targets if str(item) in VALID_CRITERIA]


def normalize_rule_result(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        entry = {}

    return RuleResult(
        rule_id=safe_text(entry.get("rule_id")) or "unknown_rule",
        category=safe_text(entry.get("category")) or "unknown",
        status=entry.get("status") if entry.get("status") in {"triggered", "not_triggered", "not_evaluated"} else "triggered",
        severity=normalize_rule_severity(entry.get("severity")),
        action=normalize_rule_action(entry.get("action")),
        blocking=bool(entry.get("blocking", False)),
        target_criteria=normalize_target_criteria(entry.get("target_criteria") or []),
        message=safe_text(entry.get("message")),
        evidence=entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {},
        score_caps={k: safe_score(v) for k, v in (entry.get("score_caps") or {}).items() if safe_score(v) is not None},
        penalties=entry.get("penalties") if isinstance(entry.get("penalties"), list) else [],
        suggestion=safe_text(entry.get("suggestion")),
    ).model_dump()


def build_validity_result_from_rules(rule_entries: List[Any], transcript_source: str, transcript_word_count: int) -> Dict[str, Any]:
    rule_results = [normalize_rule_result(entry) for entry in rule_entries]
    has_blocking_reject = any(
        r.get("blocking") and r.get("action") == "reject_or_zero"
        for r in rule_results
    )

    if has_blocking_reject:
        return ValidityResult(
            valid_for_scoring=False,
            action="reject_or_zero",
            overall_severity="critical",
            rule_results=rule_results,
            flags=[],
            score_caps={
                "pronunciation_max": 0,
                "fluency_max": 0,
                "grammar_max": 0,
                "vocabulary_max": 0,
                "coherence_max": 0,
            },
            penalties=[],
            notes=[],
            transcript_source=transcript_source,
            transcript_word_count=transcript_word_count,
        ).model_dump()

    return ValidityResult(
        valid_for_scoring=True,
        action="score",
        overall_severity="none",
        rule_results=rule_results,
        flags=[],
        score_caps={},
        penalties=[],
        notes=[],
        transcript_source=transcript_source,
        transcript_word_count=transcript_word_count,
    ).model_dump()


def build_validity_result_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    rule_entries = payload.get("rule_results") or payload.get("triggered_rules") or []
    rule_results = [normalize_rule_result(entry) for entry in rule_entries]
    action = payload.get("action") or "score"
    if action == "score_normally":
        action = "score"

    overall_severity = payload.get("overall_severity")
    if overall_severity not in VALID_SEVERITIES:
        overall_severity = "critical" if action == "reject_or_zero" else "none"

    return ValidityResult(
        valid_for_scoring=bool(payload.get("valid_for_scoring", True)),
        action=normalize_rule_action(action),
        overall_severity=overall_severity,
        rule_results=rule_results,
        flags=payload.get("flags") if isinstance(payload.get("flags"), list) else [],
        score_caps={k: safe_score(v) for k, v in (payload.get("score_caps") or {}).items() if safe_score(v) is not None},
        penalties=payload.get("penalties") if isinstance(payload.get("penalties"), list) else [],
        notes=payload.get("notes") if isinstance(payload.get("notes"), list) else [],
        transcript_source=safe_text(payload.get("transcript_source")) or "transcribed_text",
        transcript_word_count=int(payload.get("transcript_word_count") or 0),
    ).model_dump()


def build_validity_result_from_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return ValidityResult().model_dump()

    score_caps = {
        "coherence_max": safe_score(metrics.get("coherence_cap")),
        "vocabulary_max": safe_score(metrics.get("lexical_range_cap")),
        "grammar_max": safe_score(metrics.get("grammar_range_cap")),
    }
    word_count = metrics.get("word_count") or 0
    length_category = safe_text(metrics.get("length_category"))

    if length_category == "too_short":
        message = safe_text(metrics.get("note")) or safe_text(metrics.get("length_note")) or "Answer is too short for the task."
        return ValidityResult(
            valid_for_scoring=True,
            action="score_with_penalty",
            overall_severity="medium",
            rule_results=[normalize_rule_result({
                "rule_id": "answer_length.too_short",
                "category": "length",
                "severity": "medium",
                "action": "score_with_penalty",
                "blocking": False,
                "message": message,
                "evidence": {
                    "word_count": metrics.get("word_count"),
                    "expected_min_words": metrics.get("expected_min_words"),
                },
                "target_criteria": ["coherence", "vocabulary", "grammar"],
                "score_caps": score_caps,
                "penalties": [],
                "suggestion": "Expand your answer to include more details so coherence can be scored more accurately.",
            } )],
            flags=[{
                "code": "too_short",
                "severity": "medium",
                "target": "coherence",
                "message": message,
            }],
            score_caps=score_caps,
            penalties=[],
            notes=[message],
            transcript_source="transcribed_text",
            transcript_word_count=word_count,
        ).model_dump()

    penalties: List[Dict[str, Any]] = []
    if safe_text(metrics.get("score_penalty")):
        penalties.append({
            "code": "length_penalty",
            "severity": "low",
            "message": safe_text(metrics.get("score_penalty")),
        })

    return ValidityResult(
        valid_for_scoring=True,
        action="score",
        overall_severity="none",
        rule_results=[],
        flags=[],
        score_caps=score_caps,
        penalties=penalties,
        notes=[],
        transcript_source="transcribed_text",
        transcript_word_count=word_count,
    ).model_dump()
