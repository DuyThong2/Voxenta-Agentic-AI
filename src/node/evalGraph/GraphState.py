from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from node.state_models import (
    SpeakingInput,
    FormattedPronunciationResult,
)
from schemas.validity import ValidityResult
from schemas.scoring import CriterionScore


def _merge_metadata(current: Optional[Dict[str, Any]], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer for `metadata`: pronunciation_eval/answer_length_analysis/coherence_eval/
    lexical_eval/grammar_eval all run in parallel (see graphConfig.build_graph) and each
    write a DIFFERENT namespaced sub-key here (e.g. "coherence_error", "grammar_confidence")
    -- a plain dict is not enough since LangGraph raises InvalidUpdateError when more than
    one node in the same superstep writes the same key with no reducer. Dict-merging instead
    of last-write-wins is required so concurrent branches' sub-keys don't clobber each other."""
    return {**(current or {}), **(update or {})}


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    speaking_input: SpeakingInput
    pronunciation_result: FormattedPronunciationResult

    # Written by exactly one node each (answer_length_analysis / coherence_eval / lexical_eval /
    # grammar_eval respectively) -- no reducer needed since each key has a single writer, even
    # though those four nodes run in parallel (see graphConfig.build_graph). merge_scores_node
    # reads all of them plus pronunciation_result to assemble the final criteria.
    answer_length_metrics: Optional[Dict[str, Any]]
    coherence_criterion: Optional[CriterionScore]
    lexical_criterion: Optional[CriterionScore]
    grammar_criterion: Optional[CriterionScore]

    status: Literal["idle", "processing", "completed", "error"]
    error: Optional[str]
    metadata: Annotated[Dict[str, Any], _merge_metadata]
    validity: Optional[ValidityResult]
