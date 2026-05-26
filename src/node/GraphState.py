from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .state_models import (
    SpeakingInput,
    FormattedPronunciationResult,
)


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    speaking_input: SpeakingInput
    pronunciation_result: FormattedPronunciationResult

    status: Literal["idle", "processing", "completed", "error"]
    error: Optional[str]
    metadata: Dict[str, Any]