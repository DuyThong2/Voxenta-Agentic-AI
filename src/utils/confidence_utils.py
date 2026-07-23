import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils.speech_client import normalize_for_wer, word_error_rate


def clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def quality_from_snr(snr_db: Optional[float]) -> Optional[float]:
    return None if snr_db is None else clip_unit((snr_db - 5.0) / 5.0)


def quality_from_speech_ratio(silence_ratio: Optional[float]) -> Optional[float]:
    if silence_ratio is None:
        return None
    speech_ratio = 1.0 - clip_unit(silence_ratio)
    return clip_unit((speech_ratio - 0.20) / 0.20)


def compute_reference_confidence(
    raw_transcript: str,
    references: Sequence[str],
) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
    """Trả về reference medoid, C_ref, độ ổn định S và độ lệch D."""
    usable = [item.strip() for item in references if item and item.strip()]
    if len(usable) != 3:
        fallback = usable[0] if usable else raw_transcript
        return fallback, None, None, None

    # WER(A, B) != WER(B, A) in general (edit_distance / len(reference), and the two
    # candidates have different lengths) -- pairwise agreement between two arbitrary
    # rewrites has no inherent reference/hypothesis direction, so both S (pairwise
    # stability) and medoid selection must use the SAME symmetric average, not let S
    # silently depend on which of the 3 rewrites happens to have the lower list index.
    tokenized = [normalize_for_wer(item).split() for item in usable]
    pair_similarities: List[float] = []
    pair_scores = [[0.0] * 3 for _ in range(3)]
    for left in range(3):
        for right in range(left + 1, 3):
            forward = word_error_rate(tokenized[left], tokenized[right])
            reverse = word_error_rate(tokenized[right], tokenized[left])
            if forward is None or reverse is None:
                continue
            symmetric_similarity = clip_unit(1.0 - ((forward + reverse) / 2.0))
            pair_similarities.append(symmetric_similarity)
            pair_scores[left][right] = symmetric_similarity
            pair_scores[right][left] = symmetric_similarity

    if len(pair_similarities) != 3:
        return usable[0], None, None, None

    medoid_index = max(
        range(3),
        key=lambda index: sum(pair_scores[index][other] for other in range(3) if other != index),
    )
    medoid = usable[medoid_index]
    stability = sum(pair_similarities) / len(pair_similarities)
    drift = word_error_rate(
        normalize_for_wer(raw_transcript).split(),
        normalize_for_wer(medoid).split(),
    )
    if drift is None:
        return medoid, None, stability, None

    return medoid, clip_unit(min(stability, 1.0 - drift)), stability, drift


def compute_alignment_confidence(
    segments_data: Sequence[Dict[str, Any]],
    reference_text: str,
) -> Optional[float]:
    """Tính C_align = min(1-m, coverage, 1-j) từ forced-alignment của Azure."""
    reference_count = len(normalize_for_wer(reference_text).split())
    if reference_count == 0:
        return None

    words: List[Dict[str, Any]] = []
    for segment in segments_data:
        nbest = segment.get("NBest") or []
        if nbest:
            words.extend(nbest[0].get("Words") or [])

    insertions = 0
    omissions = 0
    aligned_durations: List[float] = []
    aligned_count = 0
    for word in words:
        assessment = word.get("PronunciationAssessment") or {}
        error_type = str(assessment.get("ErrorType") or "").lower()
        if error_type == "insertion":
            insertions += 1
            continue
        if error_type == "omission":
            omissions += 1
            continue

        aligned_count += 1
        duration = word.get("Duration")
        if isinstance(duration, (int, float)) and duration > 0:
            aligned_durations.append(float(duration))

    miscue_ratio = (insertions + omissions) / max(reference_count, 1)
    coverage = min(1.0, aligned_count / reference_count)

    timing_anomalies = 0
    if len(aligned_durations) >= 3:
        median_duration = median(aligned_durations)
        absolute_deviations = [abs(value - median_duration) for value in aligned_durations]
        mad = median(absolute_deviations)
        if mad > 0:
            timing_anomalies = sum(
                1 for value in aligned_durations if abs(value - median_duration) > 3.0 * mad
            )
    timing_anomaly_ratio = timing_anomalies / max(aligned_count, 1)

    return clip_unit(min(1.0 - miscue_ratio, coverage, 1.0 - timing_anomaly_ratio))


_CONSENSUS_ORDERS = (
    "accuracy, range, then task fit",
    "task fit, accuracy, then range",
    "range, task fit, then accuracy",
)
_WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)


@dataclass(frozen=True)
class ConsensusJudgment:
    response: Dict[str, Any]
    confidence: float
    score_delta_on_ten_point_scale: float


def _rationale_is_grounded(note: Any, transcript: str) -> bool:
    if not isinstance(note, str) or not note.strip():
        return False
    transcript_tokens = {
        token.lower()
        for token in _WORD_PATTERN.findall(transcript)
        if len(token) > 2
    }
    note_tokens = {token.lower() for token in _WORD_PATTERN.findall(note)}
    return not transcript_tokens or bool(transcript_tokens & note_tokens)


def run_consensus_judgment(
    call_llm: Callable[[str, str], Dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    transcript: str,
    *,
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> ConsensusJudgment:
    """Chạy ba lượt độc lập song song và lấy median score cùng C_LLM,c."""
    score_span = score_max - score_min
    if not math.isfinite(score_span) or score_span <= 0:
        score_span = 100.0

    def run_one(index: int) -> Dict[str, Any]:
        variation = (
            "\n\n## Independent calibration pass\n"
            f"Consider evidence in this order: {_CONSENSUS_ORDERS[index]}. "
            "Keep the same rubric and score scale. In the note, cite at least one exact "
            "word or phrase from the student's transcript as evidence."
        )
        return call_llm(system_prompt, user_prompt + variation)

    responses: List[Dict[str, Any]] = []
    invalid_output = False
    first_error: Optional[Exception] = None
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, index) for index in range(3)]
        for future in as_completed(futures):
            try:
                response = future.result()
                score = float(response["score"])
                if (
                    not math.isfinite(score)
                    or score < score_min
                    or score > score_max
                    or not _rationale_is_grounded(response.get("note"), transcript)
                ):
                    invalid_output = True
                    continue
                responses.append(response)
            except Exception as exc:
                invalid_output = True
                first_error = first_error or exc

    if not responses:
        if first_error is not None:
            raise first_error
        raise ValueError("Không có lượt chấm LLM hợp lệ")

    ordered = sorted(responses, key=lambda item: float(item["score"]))
    selected = ordered[len(ordered) // 2]
    if len(responses) < 3:
        return ConsensusJudgment(selected, 0.0, 0.0)

    scores = [float(item["score"]) for item in responses]
    delta_on_ten_point_scale = (max(scores) - min(scores)) * 10.0 / score_span
    confidence = 0.0 if invalid_output else clip_unit(1.0 - delta_on_ten_point_scale / 2.0)
    return ConsensusJudgment(selected, confidence, delta_on_ten_point_scale)
