import logging
from collections.abc import Iterable
from typing import Any


# Taxonomy dong, 4 nhan (truoc day 13). Day la chot LOC THAT SU: nhan nao LLM tra ve ma
# khong nam trong tap nay se bi bo ngay sau khi cham.
#
# PHAI khop tung chu voi SubAttributePolicy.java va
# node/questionGenerationGraph/constants.py::ALLOWED_SUB_ATTRIBUTES.
#
# vocabulary khong con nhan nao: 3 nhan cu (limited_range, repetition, weak_collocation) do
# duoc nhung khong ra de nham trung duoc, nen chi luyen o muc tieu chi.
ALLOWED_WEAKNESS_LABELS = {
    "grammar": {
        "tense_control",
        "complex_clause_control",
    },
    "vocabulary": set(),
    "coherence": {
        "weak_progression",
        "limited_support",
    },
}

MAX_EVIDENCE_SPANS = 5
MAX_EVIDENCE_SPAN_LENGTH = 200


def sanitize_criterion_diagnostics(
    item: dict[str, Any],
    criterion_key: str,
    *,
    allowed_band_codes: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    allowed_labels = ALLOWED_WEAKNESS_LABELS.get(criterion_key, set())
    labels: list[str] = []
    for raw_label in item.get("weakness_labels") or []:
        label = str(raw_label).strip()
        if label in allowed_labels:
            labels.append(label)
        elif label:
            log.warning(
                "Dropped weakness label outside taxonomy criterion=%s label=%s",
                criterion_key,
                label,
            )

    evidence_spans = [
        str(value)[:MAX_EVIDENCE_SPAN_LENGTH]
        for value in (item.get("evidence_spans") or [])[:MAX_EVIDENCE_SPANS]
        if value is not None
    ]

    matched_band_code = str(item.get("matched_band_code") or "").strip()
    if allowed_band_codes is not None:
        allowed_codes = {str(code) for code in allowed_band_codes if code}
        if matched_band_code and matched_band_code not in allowed_codes:
            log.warning(
                "Dropped matched band outside framework criterion=%s band=%s",
                criterion_key,
                matched_band_code,
            )
            matched_band_code = ""

    return {
        "weakness_labels": labels,
        "evidence_spans": evidence_spans,
        "recommendation_tag": str(item.get("recommendation_tag") or ""),
        "matched_band_code": matched_band_code,
    }
