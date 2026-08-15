import logging
import os

from node.evalGraph.GraphState import GraphState
from schemas.enums import SpeakingMode
from utils.speech_client import normalize_text, transcribe_with_confidence

logger = logging.getLogger(__name__)


def start_node(state: GraphState) -> dict:
    speaking_input = state.get("speaking_input")

    if speaking_input is None:
        return {
            **state,
            "status": "error",
            "error": "speaking_input is required for start_node",
        }

    answer_id = getattr(speaking_input, "answer_id", None)
    turn_order = (state.get("metadata") or {}).get("turn_order")

    audio_path = speaking_input.audio_path
    # Thiếu audio KHÔNG còn là lỗi cứng, miễn là đã có sẵn transcript.
    #
    # Bên thi giữ lại lượt nói dù bản ghi âm không tới (vox ab7bc04), nên audio_url có thể
    # rỗng trong khi Voice Live vẫn kịp ghi transcript realtime. Trả status="error" ở đây
    # làm hỏng graph.invoke() của lượt đó, kéo theo _evaluate_turn ném, và cả bài thi không
    # chấm được -- mất luôn cả những lượt hoàn toàn bình thường.
    #
    # Không có audio thì chỉ mất phần PHÁT ÂM (pronunciation_eval_node tự trả
    # pronunciation_error khi audio_path rỗng, không làm hỏng graph). Ngữ pháp, từ vựng,
    # mạch lạc, độ dài đều chấm được từ transcript.
    #
    # Vẫn là lỗi khi KHÔNG có cả hai: không audio để phiên âm, cũng không transcript sẵn --
    # lúc đó thật sự không còn gì để chấm.
    audio_missing = not audio_path or not os.path.exists(audio_path)
    if audio_missing:
        logger.warning(
            "[eval:start] khong co audio, cham theo transcript va bo qua phat am "
            "answer_id=%s turn=%s audio_path=%s co_transcript=%s",
            answer_id, turn_order, audio_path,
            bool(speaking_input.realtime_transcript),
        )
        # Xoá đường dẫn không dùng được để các node sau nhìn vào là biết ngay không có audio,
        # thay vì cầm một đường dẫn trỏ tới file không tồn tại.
        speaking_input.audio_path = None
        audio_path = None

    try:
        if speaking_input.realtime_transcript:
            # Prefer the live Voice-Live transcript (Voice Live's configured transcription
            # model, e.g. mai-transcribe-1 -- see agents/.env AZURE_VOICELIVE_TRANSCRIPTION_MODEL)
            # captured during the exam itself over re-transcribing audio_path via the Azure
            # Speech SDK -- confirmed live: Voice-Live handles code-switched Vietnamese
            # noticeably better (the Speech SDK sometimes garbles it into nonsense English,
            # e.g. "banh MI" instead of "bánh mì"). Confidence for this source comes from Voice
            # Live's own per-token logprobs (session.py's current_transcript_confidence, via
            # realtime_transcript_confidence) -- may still be None if no utterance in this turn
            # ever reported logprobs, same fallback exam_event_builder._compute_audio_quality
            # already handles.
            transcript = speaking_input.realtime_transcript
            asr_confidence = speaking_input.realtime_transcript_confidence
            logger.info(
                "[eval:start] using realtime transcript answer_id=%s turn=%s chars=%d",
                answer_id, turn_order, len(transcript),
            )
        elif audio_path is None:
            # Không audio VÀ không transcript = thí sinh im lặng thật (hoặc bản ghi âm không
            # tới). Trả transcript rỗng thay vì ném: đúng nguyên tắc đã ghi ở khối chú thích
            # ngay dưới -- validity_node có sẵn luật "audio.no_speech" biến transcript rỗng
            # thành reject_or_zero, tức lượt vẫn được ghi nhận và chấm 0, thay vì làm hỏng
            # cả bài. Ném ở đây là tái lập đúng vụ kẹt hàng đợi 2026-08-15.
            transcript, asr_confidence = "", None
            logger.info(
                "[eval:start] khong audio va khong transcript -- coi nhu im lang, cham 0 "
                "answer_id=%s turn=%s",
                answer_id, turn_order,
            )
        else:
            logger.info("[eval:start] transcribing answer_id=%s turn=%s audio_path=%s", answer_id, turn_order, audio_path)
            transcript, asr_confidence = transcribe_with_confidence(audio_path, speaking_input.language)
            # No usable speech recognized (genuine silence, or every segment rejected by ASR)
            # is NOT treated as a hard error here -- returning status="error" would abort this
            # turn's graph.invoke() with an exception, which in the multi-turn exam consumer
            # (_evaluate_turn) crashes the ENTIRE answer's evaluation (all turns, not just this
            # one), publishing evaluation_failed with no score AND no transcript for any turn.
            # validity_node's own "audio.no_speech" rule already exists to classify an empty
            # transcript as reject_or_zero (scored 0, but still a complete, recorded turn with
            # whatever transcript exists) -- let it reach that rule instead of dying here. A
            # genuine Azure API failure (network/auth/timeout) still raises inside
            # transcribe_with_confidence and is caught by the except block below, unchanged.
            transcript = transcript or ""

        if speaking_input.mode == SpeakingMode.SCRIPTED:
            # For scripted mode, reference_text is authoritative and should be normalized.
            speaking_input.reference_text = normalize_text(speaking_input.reference_text)

        speaking_input.transcribed_text = transcript
        speaking_input.asr_confidence = asr_confidence

        logger.info(
            "[eval:start] transcribed answer_id=%s turn=%s chars=%d",
            answer_id, turn_order, len(transcript),
        )

        return {
            **state,
            "speaking_input": speaking_input,
            "status": "processing",
            "metadata": {
                **state.get("metadata", {}),
                "transcription_text": transcript,
            },
        }

    except Exception as exc:
        logger.exception("[eval:start] failed answer_id=%s turn=%s", answer_id, turn_order)
        return {
            **state,
            "status": "error",
            "error": str(exc),
        }
