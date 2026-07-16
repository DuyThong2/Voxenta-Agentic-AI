"""Fan-in node: combines the 4 parallel branches (pronunciation_eval,
coherence_eval, lexical_eval, grammar_eval -- see graphConfig.build_graph)
into one final pronunciation_result and reports the turn's overall
status/error.

None of those 4 branches write the shared "status"/"error" keys while
running in parallel (that would conflict -- see GraphState._merge_metadata's
docstring); each instead stashes its own outcome under a namespaced
metadata key ("pronunciation_error", "coherence_error", etc.), only set on
failure. This node is the single place that reads all of them, so it's the
only node allowed to set "status"/"error" for this part of the graph.
"""

from typing import Any, Dict

_BRANCH_ERROR_KEYS = (
    "pronunciation_error",
    "answer_length_error",
    "coherence_error",
    "lexical_error",
    "grammar_error",
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
