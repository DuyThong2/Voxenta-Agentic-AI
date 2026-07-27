"""Fan-in node for pronunciation and combined language-quality scoring.

Concurrent branches only write namespaced metadata. This node is the single
place that sets the shared status/error and attaches language criteria to the
pronunciation result.
"""

from typing import Any, Dict

_BRANCH_ERROR_KEYS = (
    "pronunciation_error",
    "azure_score_scale_error",
    "answer_length_error",
    "language_quality_error",
)


def merge_scores_node(state: Dict[str, Any]) -> Dict[str, Any]:
    metadata = state.get("metadata") or {}
    branch_errors = {key: metadata[key] for key in _BRANCH_ERROR_KEYS if metadata.get(key)}

    pronunciation_result = state.get("pronunciation_result")
    if pronunciation_result is None:
        branch_errors.setdefault("pronunciation_error", "pronunciation_result missing at merge_scores")

    if branch_errors:
        combined = "; ".join(f"{key}: {message}" for key, message in branch_errors.items())
        return {"status": "error", "error": combined}

    coherence_criterion = state.get("coherence_criterion")
    lexical_criterion = state.get("lexical_criterion")
    grammar_criterion = state.get("grammar_criterion")

    if coherence_criterion is not None:
        pronunciation_result.criteria.coherence = coherence_criterion
    if lexical_criterion is not None:
        pronunciation_result.criteria.vocabulary = lexical_criterion
    if grammar_criterion is not None:
        pronunciation_result.criteria.grammar = grammar_criterion

    return {
        "pronunciation_result": pronunciation_result,
        "status": "completed",
        "error": None,
    }
