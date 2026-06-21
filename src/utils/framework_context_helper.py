from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node.state_models import SpeakingInput


def build_framework_criterion_context(speaking_input: "SpeakingInput", criterion_key: str) -> str:
    """Build the scoring-framework block for a criterion, or return empty string."""
    match = next(
        (
            cf
            for cf in (speaking_input.criteria_frameworks or [])
            if cf.criterion_key == criterion_key
        ),
        None,
    )
    if match is None:
        return ""

    lines = ["## Scoring Framework"]
    if match.framework_criterion_name:
        lines.append(f"Framework criterion: {match.framework_criterion_name}")
    if match.framework_criterion_description:
        lines.append(f"Definition: {match.framework_criterion_description}")
    lines.append(
        f"Score range for this criterion: {match.rubric_min_score}-{match.rubric_max_score}"
    )

    for band in sorted(match.bands, key=lambda b: b.score_min):
        line = f"- {band.code} ({band.score_min}-{band.score_max})"
        if band.label:
            line += f" [{band.label}]"
        if band.descriptor:
            line += f": {band.descriptor}"
        lines.append(line)
        if band.positive_signals:
            lines.append(f"  Positive signals: {', '.join(band.positive_signals)}")
        if band.negative_signals:
            lines.append(f"  Negative signals: {', '.join(band.negative_signals)}")

    return "\n".join(lines)
