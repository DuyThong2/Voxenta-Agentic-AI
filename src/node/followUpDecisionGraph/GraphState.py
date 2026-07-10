from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from node.state_models import QuestionContext


class FollowUpGraphState(TypedDict, total=False):
    answer_id: str
    audio_ref: str
    paper_item_id: Optional[str]
    question: Optional[QuestionContext]
    language: str
    audio_path: str
    turn_order: int
    prompt_text: Optional[str]
    active_prompt_text: Optional[str]
    current_turn: Dict[str, Any]
    turns: Annotated[List[Dict[str, Any]], add]
    published_turn_orders: Annotated[List[int], add]
    signals: Dict[str, Any]
    edge_case_handled: bool
    decision: Dict[str, Any]
    status: Literal["idle", "processing", "completed", "error"]
    error: Optional[str]
