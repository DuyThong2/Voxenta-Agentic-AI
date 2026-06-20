from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class FollowUpGraphState(TypedDict, total=False):
    answer_id: str
    question_text: Optional[str]
    evaluation_guide: Any
    language: str
    audio_path: str
    current_turn_order: int
    current_prompt_text: Optional[str]
    current_turn: Dict[str, Any]
    turns: Annotated[List[Dict[str, Any]], add]
    follow_up_count: int
    decision: Dict[str, Any]
    status: Literal["idle", "processing", "completed", "error"]
    error: Optional[str]
