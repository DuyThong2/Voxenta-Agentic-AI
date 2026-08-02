import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from node.gradingDiagnosticsGraph.constants import MAX_EVIDENCE_SPAN_LENGTH, MODEL
from node.gradingDiagnosticsGraph.prompt import build_grading_diagnostics_prompt
from schemas.grading_diagnostics import (
    GradingDiagnosticsItemRequest,
    GradingDiagnosticsItemResult,
    GradingDiagnosticsLabel,
    GradingDiagnosticsRequest,
    GradingDiagnosticsResponse,
)
from utils.criterion_diagnostics import ALLOWED_WEAKNESS_LABELS

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


class _CriterionLabels(BaseModel):
    criterion_code: str
    labels: list[str] = Field(default_factory=list)
    evidence_span: str = ""


class _DiagnosisResult(BaseModel):
    criteria: list[_CriterionLabels] = Field(default_factory=list)


def infer_grading_diagnostics(request: GradingDiagnosticsRequest) -> GradingDiagnosticsResponse:
    return GradingDiagnosticsResponse(
        items=[_infer_for_item(item) for item in request.items]
    )


def _infer_for_item(item: GradingDiagnosticsItemRequest) -> GradingDiagnosticsItemResult:
    # Chi tieu chi co taxonomy nhan text (grammar/vocabulary/coherence) moi suy luan duoc --
    # pronunciation/fluency van chi den tu tin hieu audio cua nhanh AI, feedback text khong du
    # can cu (xem WeaknessObservationDerivationService ben Java).
    applicable_criteria = [c for c in item.criteria if c in ALLOWED_WEAKNESS_LABELS]
    if not applicable_criteria or not item.feedback_summary.strip():
        return GradingDiagnosticsItemResult(item_id=item.item_id, labels=[])

    try:
        result = _diagnose(item.feedback_summary, applicable_criteria)
    except Exception:
        logger.exception(
            "Grading diagnostics inference failed for item_id=%s; skipping", item.item_id
        )
        return GradingDiagnosticsItemResult(item_id=item.item_id, labels=[])

    labels: list[GradingDiagnosticsLabel] = []
    for entry in result.criteria:
        allowed = ALLOWED_WEAKNESS_LABELS.get(entry.criterion_code)
        if allowed is None:
            continue
        evidence = entry.evidence_span[:MAX_EVIDENCE_SPAN_LENGTH]
        for raw_label in entry.labels:
            label = raw_label.strip()
            if label in allowed:
                labels.append(GradingDiagnosticsLabel(
                    criterion_code=entry.criterion_code,
                    label=label,
                    evidence_span=evidence,
                ))
            elif label:
                logger.warning(
                    "Dropped grading-diagnostics label outside taxonomy criterion=%s label=%s",
                    entry.criterion_code,
                    label,
                )
    return GradingDiagnosticsItemResult(item_id=item.item_id, labels=labels)


def _diagnose(feedback_summary: str, applicable_criteria: list[str]) -> _DiagnosisResult:
    response = _openai_client().responses.parse(
        model=MODEL,
        reasoning={"effort": "low"},
        input=[
            {
                "role": "system",
                "content": (
                    "You classify a teacher's grading note against a fixed weakness-label "
                    "taxonomy. Only select labels the note clearly supports."
                ),
            },
            {
                "role": "user",
                "content": build_grading_diagnostics_prompt(feedback_summary, applicable_criteria),
            },
        ],
        text_format=_DiagnosisResult,
    )
    return response.output_parsed or _DiagnosisResult(criteria=[])


def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client
