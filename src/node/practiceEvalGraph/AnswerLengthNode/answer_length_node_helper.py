"""Helper functions for the answer-length analysis node.

Deliberately duplicated (not shared) from the eval-node helpers: this node
only needs question-context building and transcript selection, not the
framework-context or LLM-eval-runner pieces the other eval nodes use.
"""

from typing import Optional, Tuple

from node.state_models import SpeakingInput
from schemas.enums import SpeakingMode


def _format_asset_context(speaking_input: SpeakingInput) -> list[str]:
    q = speaking_input.question
    if not q or not q.asset:
        return []

    asset = q.asset
    parts = ["\n## Question Asset"]
    if asset.type:
        parts.append(f"Asset type: {asset.type}")
    # Lay TAT CA cac truong co gia tri, khong phai cai dau tien.
    #
    # Truoc day day la chuoi if/elif nen voi AUDIO/VIDEO -- loai luon co transcript -- thi
    # description bi bo qua hoan toan. Nguoi soan de van dien no vi form co o do, roi khong hieu
    # sao AI khong biet gi ve boi canh minh da mo ta. Hai truong noi hai chuyen khac nhau:
    # transcript la NOI DUNG (loi thoai / doan van), description la BOI CANH (canh gi, giong ai,
    # dinh dang ra sao). Ca hai deu co ich.
    if asset.transcript:
        parts.append(f"Asset transcript: {asset.transcript}")
    if asset.description:
        parts.append(f"Asset description: {asset.description}")
    if asset.alt_text:
        parts.append(f"Asset alt text: {asset.alt_text}")
    if asset.transcript or asset.description or asset.alt_text:
        parts.append(
            "(Note: this is objective/factual information about the asset's content, "
            "NOT a model answer or the single correct interpretation. If the question "
            "asks the student to describe feelings, meaning, or their opinion about the "
            "asset, a DIFFERENT interpretation than what's described above is still VALID "
            "as long as it genuinely engages with the asset's actual content and is "
            "reasonably argued -- do not mark it off-topic or incorrect just because it "
            "diverges from this description.)"
        )
    # Ban sao co chu dich cua evalGraph/AnswerLengthNode -- sua o day phai sua ca hai.
    if (asset.type or "").upper() == "TEXT_PASSAGE":
        parts.append(
            "IMPORTANT -- this asset is a TEXT_PASSAGE the learner could read on screen while "
            "speaking. When judging whether the answer is long enough to assess, count only what "
            "the learner produced themselves: an answer that is mostly the passage read back "
            "aloud is NOT a substantial answer no matter how many words it contains."
        )
    return parts


def select_text_for_language_scoring(
    speaking_input: SpeakingInput,
) -> Tuple[Optional[str], str]:
    """Pick the best transcript for LLM-based language scoring.

    conversation_transcript (the timestamped AI/User dialogue) is deliberately
    never selected here -- it includes the AI's own prompt/follow-up text,
    which would inflate word_count/sentence_count. The combined language-quality
    node adds it back only as context for its coherence judgment.

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

    if mode == SpeakingMode.SCRIPTED and source != "transcribed_text":
        meta["language_scoring_note"] = (
            "Scripted mode: LLM score is diagnostic only. "
            "Official scoring uses Azure pronunciation assessment."
        )
        meta["language_scoring_status"] = "diagnostic_only"

    return meta


def build_question_context(speaking_input: SpeakingInput) -> str:
    """Build question/topic context block for the LLM prompt."""
    parts = []
    q = speaking_input.question
    t = speaking_input.topic

    if q and q.question_text:
        parts.append(f'Question: "{q.question_text}"')
    if q and q.question_type:
        # .value: tu Python 3.11, dinh dang Enum co mixin str tra ve "QuestionType.OPINION"
        # chu khong phai "opinion" -- ma prompt neu luat theo dung cac gia tri viet thuong.
        parts.append(f"Question type: {getattr(q.question_type, 'value', q.question_type)}")
    if q and q.difficulty_level:
        parts.append(f"Difficulty: {q.difficulty_level}")
    if q and q.duration_seconds is not None:
        parts.append(f"Expected duration: {q.duration_seconds}s")
    if q and q.min_response_seconds is not None and q.max_response_seconds is not None:
        parts.append(
            f"Expected response length: {q.min_response_seconds}-{q.max_response_seconds}s"
        )
    elif q and q.min_response_seconds is not None:
        parts.append(f"Expected response length: at least {q.min_response_seconds}s")
    elif q and q.max_response_seconds is not None:
        parts.append(f"Expected response length: up to {q.max_response_seconds}s")
    if t and t.topic_name:
        parts.append(f"Topic: {t.topic_name}")
    if t and t.topic_description:
        parts.append(f"Topic description: {t.topic_description}")
    if q and q.evaluation_guide:
        g = q.evaluation_guide
        parts.append("\n## Evaluation Guide")
        if g.expected_content:
            parts.append(f"Expected content: {g.expected_content}")
        if g.key_points:
            parts.append(f"Key points: {g.key_points}")
        if g.acceptable_responses:
            parts.append(f"Acceptable responses: {g.acceptable_responses}")
        if g.off_topic_examples:
            parts.append(f"Off-topic examples: {g.off_topic_examples}")
        if g.scoring_hints:
            parts.append(f"Scoring hints: {g.scoring_hints}")
        if g.common_mistakes:
            parts.append(f"Common mistakes for this question (supporting context, not exhaustive): {g.common_mistakes}")
    parts.extend(_format_asset_context(speaking_input))

    return "\n".join(parts) if parts else "No question context provided."
