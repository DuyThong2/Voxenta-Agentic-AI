"""Per-turn idempotent Kafka publishing for the realtime pipeline (Phase 2 of
docs/realtime-self-hosted-avatar-plan.md).

Extracted from the old controller/tavus_controller.py's
_wait_for_archived_turns/_publish_archived_turns/_build_answer_turn_payload,
which only ran once per *question* (at should_continue=False). This module
is redesigned to run once per *turn*: RealtimeExamSession kicks this off as a
fire-and-forget background task right after decide_next_step returns, for
every turn (not just the last one), and must never block Path A (the live
decision response to the client) on Path B (archive catch-up + Kafka).

Durability: "already published" is tracked in the same Postgres-checkpointed
state infra.database.archive_store already maintains for the `turns` list
(thread_id = answer_id) — see FollowUpGraphState.published_turn_orders in
node/followUpDecisionGraph/GraphState.py. This is intentionally not an
in-memory Python set: a process restart or a RealtimeExamSession recreated
after a reconnect must still know exactly which turns were already
published, since that's the whole point of making this durable.
"""

import logging

from events import AnswerTurnPayload, AnswerTurnsRecordedEvent, AnswerTurnsRecordedPayload
from infra.database.archive_store import aget_state, aupdate_state, archive_config, wait_for_turn
from infra.message_broker.publishers.exam_publisher import publish_answer_turns_recorded
from utils.jsonl_logger import append_jsonl

logger = logging.getLogger(__name__)

FOLLOWUP_KAFKA_LOG_FILE = "followup_kafka_publish.jsonl"


def _build_answer_turn_payload(
    turn: dict,
    answer_id: str,
    exam_attempt_id: str | None = None,
    *,
    decision_reason: str | None = None,
    realtime_transcript: str | None = None,
) -> AnswerTurnPayload:
    """`turn` đến từ POST /turns/archive, nơi transcribe_turn_node phiên âm LẠI file WAV bằng
    Azure Speech. Bản phiên âm lại đó có thể rỗng (đo được 2026-08-09: cả 3 lượt của một phiên
    thi thật đều rỗng, trong khi Voice Live đã nghe đúng nguyên câu) -- và khi đó nó ghi đè lên
    thứ duy nhất đọc được.

    Bản Voice Live nằm ở kênh `realtime_transcripts`, và tầng CHẤM đã ưu tiên nó từ trước
    (exam_consumer.get_realtime_transcript -> start_node_config). Nhưng tầng PUBLISH thì không,
    nên Java lưu chuỗi rỗng vào exam_item_responses/turns: giáo viên mở bài ra không thấy chữ
    nào, và mọi thứ đọc transcript từ DB đều tưởng thí sinh chưa nói gì.

    Chỉ thay khi bản phiên âm lại RỖNG -- có chữ thì giữ nguyên, để không đổi hành vi ở những
    lượt vốn đã chạy đúng.
    """
    transcript = turn.get("transcript") or ""
    word_count = turn.get("word_count")
    if not transcript.strip() and realtime_transcript and realtime_transcript.strip():
        transcript = realtime_transcript
        # word_count phải tính lại theo transcript MỚI. Giữ số 0 cũ sẽ đẩy một câu trả lời dài
        # xuống nhánh "câu ngắn" (isShortAnswer = wordCount < 35) và làm lệch cả ngưỡng đánh giá
        # độ ổn định lẫn các chỉ số độ dài câu trả lời.
        word_count = len(transcript.split())

    return AnswerTurnPayload(
        answer_id=turn.get("answer_id") or answer_id,
        session_id=exam_attempt_id,
        paper_item_id=turn.get("paper_item_id"),
        turn_order=turn.get("turn_order", 0),
        turn_type=turn.get("turn_type"),
        prompt_text=turn.get("prompt_text"),
        audio_url=turn.get("audio_url"),
        transcript=transcript,
        duration_seconds=turn.get("duration_seconds"),
        word_count=word_count,
        answered_at=turn.get("answered_at"),
        decision_reason=decision_reason,
    )


async def publish_turn_if_new(
    archive_graph,
    answer_id: str,
    turn_order: int,
    reason: str = "",
    exam_attempt_id: str | None = None,
    active_prompt_text: str | None = None,
    should_continue: bool = False,
    next_prompt_text: str | None = None,
) -> None:
    """Wait for this specific turn to be archived, then publish it to Kafka
    exactly once, durably. Safe to call multiple times for the same
    (answer_id, turn_order) — e.g. once per decide_next_step call and again
    on a retried /turns/archive — because the "already published" check and
    the marker write both go through archive_graph's Postgres checkpoint,
    not a process-local set.

    Intended to be run as a fire-and-forget background task
    (asyncio.create_task) by the caller — never awaited inline with the
    decision response.
    """
    config = archive_config(answer_id)

    # Persist decision_reason + the pending follow-up prompt IMMEDIATELY,
    # decoupled from the Kafka-publish path below -- that path waits (up to
    # ~90s, see archive_store._ARCHIVE_CATCHUP_RETRY_DELAYS_SECONDS) for WPF's
    # own separate POST /turns/archive (audio upload + STT) to land before it
    # does anything durable. A reconnect can legitimately happen well before
    # that slower upload completes, and RealtimeExamSession.hydrate_from_archive/
    # create_from_archive need this to resume correctly regardless of
    # whether/when the audio archive itself catches up.
    try:
        await aupdate_state(archive_graph, config, {
            "decision_reasons": [{
                "turn_order": turn_order,
                "reason": reason,
                "should_continue": should_continue,
                "next_prompt_text": next_prompt_text,
            }],
            "active_prompt_text": active_prompt_text,
        })
    except Exception:
        logger.exception(
            "[turn_publisher] failed to persist decision_reason/active_prompt_text: answer_id=%s turn_order=%d",
            answer_id, turn_order,
        )

    try:
        state_before = await aget_state(archive_graph, config)
        published_already = set((state_before.values or {}).get("published_turn_orders") or [])
        if turn_order in published_already:
            logger.debug(
                "[turn_publisher] turn already published, skipping: answer_id=%s turn_order=%d",
                answer_id, turn_order,
            )
            return

        turn = await wait_for_turn(archive_graph, answer_id, turn_order)
        if turn is None:
            logger.error(
                "[turn_publisher] giving up waiting for archived turn: answer_id=%s turn_order=%d",
                answer_id, turn_order,
            )
            return

        # Re-check right before publishing: another publish_turn_if_new call for the
        # same turn could have completed while we were polling above.
        state_now = await aget_state(archive_graph, config)
        published_now = set((state_now.values or {}).get("published_turn_orders") or [])
        if turn_order in published_now:
            logger.debug(
                "[turn_publisher] turn published concurrently, skipping: answer_id=%s turn_order=%d",
                answer_id, turn_order,
            )
            return

        # Lấy từ state_now đã đọc ngay trên -- không tốn thêm một vòng Postgres nào.
        realtime_entry = next(
            (
                entry
                for entry in ((state_now.values or {}).get("realtime_transcripts") or [])
                if entry.get("turn_order") == turn_order
            ),
            None,
        )
        event = AnswerTurnsRecordedEvent(
            answer_id=answer_id,
            payload=AnswerTurnsRecordedPayload(
                turns=[_build_answer_turn_payload(
                    turn,
                    answer_id,
                    exam_attempt_id,
                    decision_reason=reason,
                    realtime_transcript=None if realtime_entry is None else realtime_entry.get("text"),
                )],
                reason=reason,
            ),
        )
        append_jsonl(FOLLOWUP_KAFKA_LOG_FILE, {
            "answer_id": answer_id,
            "turn_order": turn_order,
            "event": event.model_dump(by_alias=True),
        })
        await publish_answer_turns_recorded(event)

        # Persist the "published" marker durably via the same checkpointer
        # archive_graph already uses, through the add reducer on
        # published_turn_orders (mirrors how append_turn_node appends to
        # turns) — not a Python-process-local set, so this survives a
        # process restart or a RealtimeExamSession recreated after reconnect.
        # (decision_reasons/active_prompt_text are persisted earlier, up
        # front — see the comment above — not here.)
        await aupdate_state(archive_graph, config, {"published_turn_orders": [turn_order]})
    except Exception:
        logger.exception(
            "[turn_publisher] failed to publish turn: answer_id=%s turn_order=%d",
            answer_id, turn_order,
        )
