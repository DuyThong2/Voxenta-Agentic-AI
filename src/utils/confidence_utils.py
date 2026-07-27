import json
import logging
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import RateLimitError as OpenAIRateLimitError

_RateLimitErrors = (OpenAIRateLimitError, AnthropicRateLimitError)

from utils.speech_client import normalize_for_wer, word_error_rate
from utils.jsonl_logger import append_jsonl

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Case (5) (Grammar/Vocabulary/Coherence) và case (3) (pronunciation reference) đều gọi LLM 3
# lần độc lập song song qua ThreadPoolExecutor(max_workers=3) -- đúng 3 worker cho đúng 3 lượt,
# không có việc nào khác chờ dùng chung pool đó, nên retry/backoff bên dưới chỉ chặn ĐÚNG 1
# thread đang giữ lượt gặp lỗi, 2 lượt song song còn lại không bị ảnh hưởng và KHÔNG cần tạo
# thêm thread nào cho việc retry/fallback -- toàn bộ vẫn nằm trong "async ở tầng ngoài"
# (exam_consumer.py bọc graph.invoke bằng asyncio.to_thread) đã có sẵn từ trước.
_RETRY_BACKOFF_SECONDS = 1.0
_OPENAI_MODEL = "gpt-5.4"
CLAUDE_MODEL = "claude-sonnet-4-6"
_LLM_CONCURRENCY_LIMIT = max(
    1,
    int(os.getenv("EVAL_LLM_CONCURRENCY_LIMIT", "16")),
)
_llm_call_semaphore = threading.BoundedSemaphore(_LLM_CONCURRENCY_LIMIT)


@contextmanager
def llm_call_slot():
    """Giới hạn tổng số lệnh LLM của evalGraph chạy đồng thời."""
    _llm_call_semaphore.acquire()
    try:
        yield
    finally:
        _llm_call_semaphore.release()


def _invoke_llm_json(llm: Any, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    with llm_call_slot():
        response = llm.invoke(messages)
    content = response.content.strip()
    if content.startswith("```"):
        lines = [line for line in content.split("\n") if not line.strip().startswith("```")]
        content = "\n".join(lines).strip()
    return json.loads(content)


def call_llm_openai(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Gọi GPT-5.4, parse JSON response."""
    return _invoke_llm_json(
        ChatOpenAI(model=_OPENAI_MODEL, reasoning_effort="medium"),
        system_prompt,
        user_prompt,
    )


def call_llm_claude(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Gọi Claude Sonnet, parse JSON response -- cần biến môi trường ANTHROPIC_API_KEY (đọc tự
    động qua langchain-anthropic, giống cách ChatOpenAI đọc OPENAI_API_KEY)."""
    return _invoke_llm_json(ChatAnthropic(model=CLAUDE_MODEL, temperature=0), system_prompt, user_prompt)


def call_with_retry_and_fallback(primary: Callable[[], _T], fallback: Callable[[], _T]) -> _T:
    """Gọi primary(); nếu bị rate-limit (429, OpenAI hoặc Anthropic tuỳ bên nào đang là primary)
    thì retry NGẮN 1 lần cùng provider trước khi đổi hẳn sang fallback() -- đa số 429 tự hết sau
    vài trăm ms tới vài giây, retry ngắn thường đã đủ; lỗi khác (không phải rate-limit) bỏ qua
    retry, đổi fallback ngay. Nếu fallback() cũng lỗi, exception đó lọt lên caller xử lý (không
    nuốt) -- caller (VD run_consensus_judgment) đã có sẵn cơ chế coi lỗi đó là 1 lượt chấm thất
    bại.

    primary/fallback là closure 0-tham số (dùng lambda/partial bind sẵn args) để hàm này dùng
    chung được cho cả case (5) (trả Dict) lẫn case (3) (trả str), không cần biết chữ ký gốc.

    Mọi lần retry/đổi-fallback đều được log (WARNING) và fallback thành công/thất bại cũng được
    log -- trước đây toàn bộ nhánh này nuốt exception im lặng nên không thể biết từ log là 429
    có xảy ra hay fallback Anthropic có chạy/thành công hay không."""
    try:
        return primary()
    except _RateLimitErrors as exc:
        logger.warning(
            "LLM primary provider bị rate-limit (%s); retry 1 lần sau %.1fs trước khi fallback",
            type(exc).__name__,
            _RETRY_BACKOFF_SECONDS,
        )
        time.sleep(_RETRY_BACKOFF_SECONDS)
        try:
            return primary()
        except Exception as retry_exc:
            logger.warning(
                "LLM primary retry vẫn lỗi (%s); đổi sang fallback provider",
                type(retry_exc).__name__,
            )
    except Exception as exc:
        logger.warning(
            "LLM primary provider lỗi (%s: %s); đổi sang fallback provider",
            type(exc).__name__,
            exc,
        )
    try:
        result = fallback()
        logger.warning("LLM fallback provider chạy THÀNH CÔNG (đã chuyển provider)")
        return result
    except Exception as fb_exc:
        logger.error(
            "LLM fallback provider CŨNG lỗi (%s: %s) -- lượt chấm này coi như thất bại",
            type(fb_exc).__name__,
            fb_exc,
        )
        raise


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


@dataclass(frozen=True)
class AlignmentConfidence:
    """3 thành phần RIÊNG của case (4), không gộp sẵn -- ngưỡng Vietnam-adjusted trong spec
    (m soft>0.20/hard>0.30, c soft<0.90/hard<0.80, j soft>0.15/hard>0.30) áp RIÊNG cho từng
    thành phần, không phải 1 ngưỡng chung cho composite min(). Gộp sớm thành 1 số duy nhất (bản
    trước) làm ngưỡng nới cho (1-m) -- đúng phần điều chỉnh Vietnam quan trọng nhất (rụng phụ âm
    cuối) -- bị vô hiệu hoá, vì mọi thành phần khi đó buộc phải đạt ngưỡng của c (0.90) mới qua
    được composite, xoá mất biên nới 0.80 dành riêng cho (1-m)."""

    accuracy: Optional[float]  # 1 - m
    coverage: Optional[float]  # c
    timing: Optional[float]  # 1 - j
    composite: Optional[float]  # min(accuracy, coverage, timing) -- CHỈ để hiển thị/cPfBranch


def compute_alignment_confidence(
    segments_data: Sequence[Dict[str, Any]],
    reference_text: str,
) -> AlignmentConfidence:
    """Tính riêng accuracy (1-m), coverage (c), timing (1-j) từ forced-alignment của Azure."""
    reference_count = len(normalize_for_wer(reference_text).split())
    if reference_count == 0:
        return AlignmentConfidence(None, None, None, None)

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

    accuracy = clip_unit(1.0 - miscue_ratio)
    coverage = clip_unit(coverage)
    timing = clip_unit(1.0 - timing_anomaly_ratio)
    composite = min(accuracy, coverage, timing)
    return AlignmentConfidence(accuracy, coverage, timing, composite)


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


# (primary, fallback) cho mỗi lượt trong 3 lượt case (5) -- xen O/C/O thay vì 3 lần cùng 1
# model/weights: nếu GPT-5.4 có 1 điểm mù/bias hệ thống nào đó, cả 3 lượt cùng model sẽ dính
# GIỐNG NHAU (Δc thấp giả tạo, trông "tự tin" dù sai đều) -- đổi 1 lượt sang model khác hẳn
# kiến trúc/dữ liệu train tăng tính độc lập thật của 3 "judge", đúng tinh thần ROVER/cross-
# system agreement đã dùng cho case (2). Đồng thời giảm tải OpenAI mỗi turn (đỡ 429).
_CONSENSUS_PROVIDERS: Tuple[Tuple[Callable[[str, str], Dict[str, Any]], Callable[[str, str], Dict[str, Any]]], ...] = (
    (call_llm_openai, call_llm_claude),
    (call_llm_claude, call_llm_openai),
    (call_llm_openai, call_llm_claude),
)

_MULTI_CRITERION_ORDERS = (
    "grammar, vocabulary, then coherence",
    "vocabulary, coherence, then grammar",
    "coherence, grammar, then vocabulary",
)


def run_consensus_judgment(
    system_prompt: str,
    user_prompt: str,
    transcript: str,
    *,
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> ConsensusJudgment:
    """Chạy ba lượt độc lập song song (xen OpenAI/Claude, mỗi lượt tự retry+fallback) và lấy
    median score cùng C_LLM,c."""
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
        primary, fallback = _CONSENSUS_PROVIDERS[index]
        prompt = user_prompt + variation
        return call_with_retry_and_fallback(
            lambda: primary(system_prompt, prompt),
            lambda: fallback(system_prompt, prompt),
        )

    # Mỗi lượt lưu (response, score, grounded). Lượt PARSE ĐƯỢC + score trong range = "có điểm",
    # dùng để tính spread; grounding tách riêng làm tín hiệu MỀM (không loại lượt).
    scored: List[Tuple[Dict[str, Any], float, bool]] = []
    first_error: Optional[Exception] = None
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, index) for index in range(3)]
        for future in as_completed(futures):
            try:
                response = future.result()
                score = float(response["score"])
                if not math.isfinite(score) or score < score_min or score > score_max:
                    # Parse được nhưng điểm vô nghĩa -> bỏ (không có số để đưa vào spread).
                    continue
                grounded = _rationale_is_grounded(response.get("note"), transcript)
                scored.append((response, score, grounded))
            except Exception as exc:
                first_error = first_error or exc

    # Ép confidence = 0 CHỈ khi thật sự < 2 lượt có điểm hợp lệ (không đủ để đo độ ổn định) --
    # KHÔNG còn ép 0 chỉ vì 1/3 lượt hỏng/ungrounded như bản cũ, theo research Vietnam-focus
    # (compass_artifact_wf-52df89c9: tách "không đủ evidence" khỏi "hệ thống bất ổn"; dùng
    # self-consistency trên các lượt hợp lệ thay vì zero-out cả tiêu chí).
    if len(scored) < 2:
        if not scored:
            if first_error is not None:
                raise first_error
            raise ValueError("Không có lượt chấm LLM hợp lệ")
        return ConsensusJudgment(scored[0][0], 0.0, 0.0)

    scored.sort(key=lambda item: item[1])
    selected = scored[len(scored) // 2][0]
    scores = [item[1] for item in scored]
    delta_on_ten_point_scale = (max(scores) - min(scores)) * 10.0 / score_span
    confidence = clip_unit(1.0 - delta_on_ten_point_scale / 2.0)
    # Grounding = PHẠT MỀM (không ép 0): nếu ĐA SỐ lượt hợp lệ có rationale không bám transcript
    # (nghi judge bịa), giảm 30%. Bản cũ chỉ cần 1 lượt ungrounded là ép cả tiêu chí về 0.
    ungrounded = sum(1 for item in scored if not item[2])
    if ungrounded * 2 > len(scored):
        confidence *= 0.7
    return ConsensusJudgment(selected, confidence, delta_on_ten_point_scale)


def run_multi_criterion_consensus_judgment(
    system_prompt: str,
    user_prompt: str,
    transcript: str,
    score_ranges: Dict[str, Tuple[float, float]],
    *,
    debug_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, ConsensusJudgment]:
    """Chạy đúng ba request O-C-O song song, mỗi request chấm toàn bộ tiêu chí.

    Việc gộp chỉ xảy ra ở tầng gọi model. Valid-run, median, spread, grounding và confidence
    vẫn được tính độc lập cho từng tiêu chí để một nhánh JSON lỗi hoặc một rubric dao động
    không làm mất kết quả đáng tin cậy của các tiêu chí còn lại.
    """
    criterion_keys = tuple(score_ranges)
    scored_by_criterion: Dict[str, List[Tuple[Dict[str, Any], float, bool]]] = {
        key: [] for key in criterion_keys
    }
    first_error: Optional[Exception] = None
    raw_rounds: List[Dict[str, Any]] = []

    def run_one(index: int) -> Tuple[int, str, Dict[str, Any]]:
        variation = (
            "\n\n## Independent calibration pass\n"
            f"Evaluate in this order: {_MULTI_CRITERION_ORDERS[index]}. "
            "Keep all three rubrics independent: never copy a score, weakness, or overall "
            "impression from one criterion into another. For every criterion, cite at least "
            "one exact word or phrase from the student's transcript in evidence_spans."
        )
        primary, fallback = _CONSENSUS_PROVIDERS[index]
        prompt = user_prompt + variation
        provider, response = call_with_retry_and_fallback(
            lambda: (primary.__name__, primary(system_prompt, prompt)),
            lambda: (fallback.__name__, fallback(system_prompt, prompt)),
        )
        return index, provider.removeprefix("call_llm_"), response

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(run_one, index): index
            for index in range(3)
        }
        for future in as_completed(futures):
            try:
                index, provider, response = future.result()
                raw_rounds.append(
                    {
                        "pass_index": index,
                        "provider": provider,
                        "response": response,
                    }
                )
            except Exception as exc:
                first_error = first_error or exc
                raw_rounds.append(
                    {
                        "pass_index": futures[future],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            for key in criterion_keys:
                criterion_response = response.get(key)
                if not isinstance(criterion_response, dict):
                    continue

                score_min, score_max = score_ranges[key]
                try:
                    score = float(criterion_response["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    not math.isfinite(score)
                    or score < score_min
                    or score > score_max
                ):
                    continue

                evidence_spans = criterion_response.get("evidence_spans")
                if isinstance(evidence_spans, list):
                    folded_transcript = transcript.casefold()
                    grounded = any(
                        isinstance(span, str)
                        and bool(span.strip())
                        and span.strip().casefold() in folded_transcript
                        for span in evidence_spans
                    )
                else:
                    grounded = _rationale_is_grounded(
                        criterion_response.get("note"),
                        transcript,
                    )
                scored_by_criterion[key].append(
                    (criterion_response, score, grounded)
                )

    append_jsonl(
        "language_quality_consensus.jsonl",
        {
            **(debug_context or {}),
            "rounds": sorted(
                raw_rounds,
                key=lambda item: item["pass_index"],
            ),
        },
    )

    judgments: Dict[str, ConsensusJudgment] = {}
    for key, scored in scored_by_criterion.items():
        if len(scored) < 2:
            if not scored:
                if first_error is not None:
                    raise first_error
                raise ValueError(f"Không có lượt chấm LLM hợp lệ cho tiêu chí {key}")
            judgments[key] = ConsensusJudgment(scored[0][0], 0.0, 0.0)
            continue

        scored.sort(key=lambda item: item[1])
        selected = scored[len(scored) // 2][0]
        scores = [item[1] for item in scored]
        score_min, score_max = score_ranges[key]
        score_span = score_max - score_min
        if not math.isfinite(score_span) or score_span <= 0:
            score_span = 100.0
        delta_on_ten_point_scale = (
            (max(scores) - min(scores)) * 10.0 / score_span
        )
        confidence = clip_unit(1.0 - delta_on_ten_point_scale / 2.0)
        ungrounded = sum(1 for item in scored if not item[2])
        if ungrounded * 2 > len(scored):
            confidence *= 0.7

        judgments[key] = ConsensusJudgment(
            selected,
            confidence,
            delta_on_ten_point_scale,
        )

    return judgments
