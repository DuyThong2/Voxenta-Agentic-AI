from typing import Optional

from schemas.common import _CamelMessage


class EvaluationGuideInput(_CamelMessage):
    expected_content: Optional[str] = None
    key_points: Optional[str] = None
    acceptable_responses: Optional[str] = None
    off_topic_examples: Optional[str] = None
    scoring_hints: Optional[str] = None
    common_mistakes: Optional[str] = None
