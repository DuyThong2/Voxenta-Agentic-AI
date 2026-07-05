from typing import Any, Dict, Iterable


def is_clarification_reason(reason: str | None) -> bool:
    return str(reason or "").startswith("clarification_")


def count_assessment_turns(turns: Iterable[Dict[str, Any]]) -> int:
    count = 0
    for turn in turns:
        reason = str((turn or {}).get("decision_reason") or "")
        if is_clarification_reason(reason):
            continue
        count += 1
    return count
