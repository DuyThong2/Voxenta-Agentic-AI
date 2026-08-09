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
    """Merge namespaced metadata emitted by concurrent graph branches."""
    return {**(current or {}), **(update or {})}


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    speaking_input: SpeakingInput
    pronunciation_result: FormattedPronunciationResult

    # The combined language_quality_eval node is the single writer for all three language
    # criteria. merge_scores_node attaches them to pronunciation_result.
    answer_length_metrics: Optional[Dict[str, Any]]
    coherence_criterion: Optional[CriterionScore]
    lexical_criterion: Optional[CriterionScore]
    grammar_criterion: Optional[CriterionScore]

    status: Literal["idle", "processing", "completed", "error"]
    error: Optional[str]
    metadata: Annotated[Dict[str, Any], _merge_metadata]
    validity: Optional[ValidityResult]
