from typing import Literal, Optional

from events.envelope import EventEnvelope
from schemas.common import _CamelMessage
from schemas.evaluation_event import EvaluationGuideInput


class QuestionAssetAnalysisRequestedPayload(_CamelMessage):
    asset_type: str
    url: Optional[str] = None
    question_text: Optional[str] = None
    evaluation_guide: Optional[EvaluationGuideInput] = None
    existing_transcript: Optional[str] = None
    existing_description: Optional[str] = None


class QuestionAssetAnalysisRequestedEvent(EventEnvelope):
    event_type: Literal["QuestionAssetAnalysisRequested"] = "QuestionAssetAnalysisRequested"

    asset_id: str
    question_id: str
    payload: QuestionAssetAnalysisRequestedPayload
