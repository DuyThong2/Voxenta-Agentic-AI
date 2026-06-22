from typing import Optional

from node.state_models import QuestionContext
from schemas.common import _CamelMessage


class FollowUpTurnRequest(_CamelMessage):
    audio_ref: str
    answer_id: str
    turn_order: int
    prompt_text: Optional[str] = None
    question: Optional[QuestionContext] = None
    language: str = "en-US"
