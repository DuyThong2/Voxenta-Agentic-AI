from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from node.state_models import QuestionContext


class FollowUpGraphState(TypedDict, total=False):
    answer_id: str
    audio_ref: str
    question: Optional[QuestionContext]
    language: str
    audio_path: str
    turn_order: int
    prompt_text: Optional[str]
    current_turn: Dict[str, Any]
    turns: Annotated[List[Dict[str, Any]], add]
    signals: Dict[str, Any]
    decision: Dict[str, Any]
    status: Literal["idle", "processing", "completed", "error"]
    error: Optional[str]
