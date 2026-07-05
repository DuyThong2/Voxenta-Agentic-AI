"""Per-turn archive catch-up + idempotent Kafka publishing for the realtime
pipeline (Phase 2 of docs/realtime-self-hosted-avatar-plan.md).

Extracted from the old controller/tavus_controller.py's
_wait_for_archived_turns/_publish_archived_turns/_build_answer_turn_payload,
which only ran once per *question* (at should_continue=False). This module
is redesigned to run once per *turn*: RealtimeExamSession kicks this off as a
fire-and-forget background task right after decide_next_step returns, for
every turn (not just the last one), and must never block Path A (the live
decision response to the client) on Path B (archive catch-up + Kafka).

Durability: "already published" is tracked in the same Postgres-checkpointed
state archive_graph already maintains for the `turns` list (thread_id =
answer_id) — see FollowUpGraphState.published_turn_orders in
node/followUpDecisionGraph/GraphState.py. This is intentionally not an
in-memory Python set: a process restart or a RealtimeExamSession recreated
after a reconnect must still know exactly which turns were already
published, since that's the whole point of making this durable.

archive_graph here is compiled with the SYNC PostgresSaver checkpointer
(app.py's app.state.archive_graph, the same instance archive_controller.py's
/turns/archive calls .invoke() on) -- NOT an AsyncPostgresSaver. Calling
.aget_state()/.aupdate_state() directly on it raises NotImplementedError
(BaseCheckpointSaver's async methods only have a real implementation when
the checkpointer itself is async-native; the sync PostgresSaver doesn't
provide one) -- confirmed via a real live run where every turn's Kafka
publish silently failed this way (caught by this module's own try/except, so
nothing crashed, it just never published). AsyncPostgresSaver was considered
and rejected: psycopg's async mode requires a SelectorEventLoop, but
asyncio's default on Windows is ProactorEventLoop, and switching the whole
app's event loop policy to fix this one path risked breaking aiortc/aioice
(used by both the proctoring and avatar WebRTC connections) for a benefit
that's achievable more simply. Instead, the sync .get_state()/.update_state()
are called via asyncio.to_thread -- same non-blocking-event-loop property,
zero new dependencies, reuses the connection pool that's already proven
working.
"""

import asyncio
import logging

from events import AnswerTurnPayload, AnswerTurnsRecordedEvent, AnswerTurnsRecordedPayload
from infra.message_broker.publishers.exam_publisher import publish_answer_turns_recorded
from utils.jsonl_logger import append_jsonl

logger = logging.getLogger(__name__)


async def _aget_state(archive_graph, config: dict):
    return await asyncio.to_thread(archive_graph.get_state, config)


async def _aupdate_state(archive_graph, config: dict, update: dict) -> None:
    await asyncio.to_thread(archive_graph.update_state, config, update)

FOLLOWUP_KAFKA_LOG_FILE = "followup_kafka_publish.jsonl"

# WPF's own /turns/archive call (S3 upload, then download-from-S3 + Azure STT) can race
# this turn's own publish attempt — the live decision path already has its transcript and
# can resolve before WPF's archive for this exact turn has landed in Postgres. Poll briefly
# for the archive to catch up before publishing instead of publishing whatever's there
# immediately and silently dropping the latest turn (this exact tuning fixed a real bug:
# question 1/3/5 all published with the last turn missing in a 2026-06-24 run) — reused
# verbatim here, do not "simplify" the timing.
_ARCHIVE_CATCHUP_RETRY_DELAYS_SECONDS = [0.3, 0.3, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]


def _build_answer_turn_payload(turn: dict, answer_id: str) -> AnswerTurnPayload:
    return AnswerTurnPayload(
        answer_id=turn.get("answer_id") or answer_id,
        turn_order=turn.get("turn_order", 0),
        turn_type=turn.get("turn_type"),
        prompt_text=turn.get("prompt_text"),
        audio_url=turn.get("audio_url"),
        transcript=turn.get("transcript", ""),
        duration_seconds=turn.get("duration_seconds"),
        word_count=turn.get("word_count"),
        answered_at=turn.get("answered_at"),
    )


def _archive_config(answer_id: str) -> dict:
    return {"configurable": {"thread_id": answer_id}}


async def _wait_for_turn(archive_graph, answer_id: str, turn_order: int) -> dict | None:
    """Poll the checkpointed `turns` list until the given turn_order shows up
    (or we give up and return None). Same retry tuning as the old
    tavus_controller._wait_for_archived_turns, just keyed on a single
    turn_order instead of a total expected_turn_count."""
    turn: dict | None = None
    for delay in [0.0, *_ARCHIVE_CATCHUP_RETRY_DELAYS_SECONDS]:
        if delay:
            await asyncio.sleep(delay)
        archived_state = await _aget_state(archive_graph, _archive_config(answer_id))
        turns = (archived_state.values or {}).get("turns") or []
        turn = next((t for t in turns if (t or {}).get("turn_order") == turn_order), None)
        if turn is not None:
            return turn

    logger.warning(
        "[turn_publisher] archive did not catch up before publish: answer_id=%s turn_order=%d",
        answer_id, turn_order,
    )
    return turn


async def get_last_archived_turn_order(archive_graph, answer_id: str) -> int:
    """Durable truth for the `resume` handshake: the highest turn_order
    actually present in the checkpointed `turns` list for this answer_id, or
    0 if none. Reads the same checkpoint state turn_publisher writes to —
    never guesses from in-memory state."""
    archived_state = await _aget_state(archive_graph, _archive_config(answer_id))
    turns = (archived_state.values or {}).get("turns") or []
    turn_orders = [int((t or {}).get("turn_order") or 0) for t in turns]
    return max(turn_orders) if turn_orders else 0


async def publish_turn_if_new(archive_graph, answer_id: str, turn_order: int, reason: str = "") -> None:
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
    config = _archive_config(answer_id)

    try:
        state_before = await _aget_state(archive_graph, config)
        published_already = set((state_before.values or {}).get("published_turn_orders") or [])
        if turn_order in published_already:
            logger.debug(
                "[turn_publisher] turn already published, skipping: answer_id=%s turn_order=%d",
                answer_id, turn_order,
            )
            return

        turn = await _wait_for_turn(archive_graph, answer_id, turn_order)
        if turn is None:
            logger.error(
                "[turn_publisher] giving up waiting for archived turn: answer_id=%s turn_order=%d",
                answer_id, turn_order,
            )
            return

        # Re-check right before publishing: another publish_turn_if_new call for the
        # same turn could have completed while we were polling above.
        state_now = await _aget_state(archive_graph, config)
        published_now = set((state_now.values or {}).get("published_turn_orders") or [])
        if turn_order in published_now:
            logger.debug(
                "[turn_publisher] turn published concurrently, skipping: answer_id=%s turn_order=%d",
                answer_id, turn_order,
            )
            return

        event = AnswerTurnsRecordedEvent(
            answer_id=answer_id,
            payload=AnswerTurnsRecordedPayload(
                turns=[_build_answer_turn_payload(turn, answer_id)],
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
        await _aupdate_state(archive_graph, config, {"published_turn_orders": [turn_order]})
    except Exception:
        logger.exception(
            "[turn_publisher] failed to publish turn: answer_id=%s turn_order=%d",
            answer_id, turn_order,
        )
