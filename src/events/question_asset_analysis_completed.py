from typing import Literal, Optional

from events.envelope import EventEnvelope
from schemas.common import _CamelMessage


class QuestionAssetAnalysisCompletedPayload(_CamelMessage):
    transcript: Optional[str] = None
    description: Optional[str] = None


class QuestionAssetAnalysisCompletedEvent(EventEnvelope):
    event_type: Literal["QuestionAssetAnalysisCompleted"] = "QuestionAssetAnalysisCompleted"

    asset_id: str
    payload: QuestionAssetAnalysisCompletedPayload
