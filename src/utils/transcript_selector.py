"""Shared helper to select the correct transcript for LLM language scoring.

LLM nodes (grammar, vocabulary, coherence, content) must evaluate based on
what the student actually said — NOT the Azure reference text or a scripted
model answer.

Priority:
  1. transcribed_text   — raw speech-to-text output, closest to what the student said.
  2. corrected_transcript — auto-corrected version (only if transcribed_text is missing).
  3. reference_text      — last-resort fallback (may be a model answer / scripted text).
"""

from typing import Optional, Tuple

from node.state_models import SpeakingInput


def select_text_for_language_scoring(
    speaking_input: SpeakingInput,
) -> Tuple[Optional[str], str]:
    """Pick the best transcript for LLM-based language scoring.

    Returns:
        (text, source) where source is one of:
          - "transcribed_text"
          - "corrected_transcript"
          - "reference_text_fallback"
          - "missing"
    """
    if speaking_input.transcribed_text:
        return speaking_input.transcribed_text, "transcribed_text"

    if speaking_input.corrected_transcript:
        return speaking_input.corrected_transcript, "corrected_transcript"

    if speaking_input.reference_text:
        return speaking_input.reference_text, "reference_text_fallback"

    return None, "missing"


def build_scoring_metadata(
    source: str,
    mode: Optional[str],
) -> dict:
    """Build metadata dict for the selected transcript source.

    Includes a diagnostic flag when the source is a reference_text fallback,
    meaning the score should NOT be treated as a real student assessment.
    """
    meta: dict = {"language_scoring_text_source": source}

    if source == "reference_text_fallback":
        meta["language_scoring_note"] = (
            "No transcribed_text or corrected_transcript available. "
            "Fell back to reference_text which may be a model answer. "
            "Score is diagnostic only, not an official assessment."
        )
        meta["language_scoring_status"] = "diagnostic_only"

    if mode == "scripted" and source != "transcribed_text":
        meta["language_scoring_note"] = (
            "Scripted mode: LLM score is diagnostic only. "
            "Official scoring uses Azure pronunciation assessment."
        )
        meta["language_scoring_status"] = "diagnostic_only"

    return meta
