from typing import Optional

from schemas.common import _CamelMessage


class FollowUpAnswerTurn(_CamelMessage):
    answer_id: Optional[str] = None
    turn_order: int
    turn_type: Optional[str] = None
    prompt_text: Optional[str] = None
    audio_url: Optional[str] = None
    transcript: str = ""
    duration_seconds: Optional[int] = None
    word_count: Optional[int] = None
    answered_at: Optional[str] = None


class FollowUpTurnResponse(_CamelMessage):
    turn_order: int
    transcript: str
    prompt_text: Optional[str] = None
    current_turn: Optional[FollowUpAnswerTurn] = None
    should_continue: bool
    next_prompt_text: Optional[str] = None
    reason: str = ""
    reached_max_turns: bool = False
