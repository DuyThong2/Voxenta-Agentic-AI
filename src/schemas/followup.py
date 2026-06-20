from typing import Optional

from schemas.common import _CamelMessage
from schemas.evaluation_event import EvaluationGuideInput


class FollowUpTurnRequest(_CamelMessage):
    answer_id: str
    turn_order: int
    question_text: Optional[str] = None
    evaluation_guide: Optional[EvaluationGuideInput] = None
    language: str = "en-US"


class FollowUpTurnResponse(_CamelMessage):
    turn_order: int
    transcript: str
    should_continue: bool
    next_prompt_text: Optional[str] = None
    reached_max_turns: bool = False
