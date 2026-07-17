"""Postgres-checkpointed read/write for per-question/per-turn realtime state
(Phase 2 of docs/realtime-self-hosted-avatar-plan.md).

Durability: all state here is tracked in the same Postgres-checkpointed state
archive_graph already maintains (thread_id = answer_id, or exam_attempt_id for
the current-question marker) — see FollowUpGraphState in
node/followUpDecisionGraph/GraphState.py. This is intentionally not in-memory
Python state: a process restart or a RealtimeExamSession recreated after a
reconnect must still be able to read it back.

archive_graph here is compiled with the SYNC PostgresSaver checkpointer
(app.py's app.state.archive_graph, the same instance archive_controller.py's
/turns/archive calls .invoke() on) -- NOT an AsyncPostgresSaver. Calling
.aget_state()/.aupdate_state() directly on it raises NotImplementedError
(BaseCheckpointSaver's async methods only have a real implementation when
the checkpointer itself is async-native; the sync PostgresSaver doesn't
provide one) -- confirmed via a real live run where every turn's Kafka
publish silently failed this way (caught by the publish workflow's own
try/except, so nothing crashed, it just never published). AsyncPostgresSaver
was considered and rejected: psycopg's async mode requires a
SelectorEventLoop, but asyncio's default on Windows is ProactorEventLoop, and
switching the whole app's event loop policy to fix this one path risked
breaking aiortc/aioice (used by both the proctoring and avatar WebRTC
connections) for a benefit that's achievable more simply. Instead, the sync
.get_state()/.update_state() are called via asyncio.to_thread -- same
non-blocking-event-loop property, zero new dependencies, reuses the
connection pool that's already proven working.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def aget_state(archive_graph, config: dict):
    return await asyncio.to_thread(archive_graph.get_state, config)


async def aupdate_state(archive_graph, config: dict, update: dict) -> None:
    await asyncio.to_thread(archive_graph.update_state, config, update)


def archive_config(answer_id: str) -> dict:
    return {"configurable": {"thread_id": answer_id}}


# WPF's own /turns/archive call (S3 upload, then download-from-S3 + Azure STT) can race
# a turn's own publish attempt — the live decision path already has its transcript and
# can resolve before WPF's archive for this exact turn has landed in Postgres. Poll briefly
# for the archive to catch up before publishing instead of publishing whatever's there
# immediately and silently dropping the latest turn (this exact tuning fixed a real bug:
# question 1/3/5 all published with the last turn missing in a 2026-06-24 run) — the first
# 8 entries are reused verbatim from that fix, do not "simplify" that part of the timing.
# Extended with a longer tail: utils.speech_client.transcribe() now uses continuous
# recognition (fixing a truncation bug in recognize_once()), so archival runs proportionally
# to real audio length instead of near-instant -- a long turn can take tens of seconds to
# actually land in Postgres, which the original ~6.6s total ceiling did not anticipate.
_ARCHIVE_CATCHUP_RETRY_DELAYS_SECONDS = [
    0.3, 0.3, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5,
    2.0, 3.0, 5.0, 5.0, 10.0, 10.0, 15.0, 15.0, 20.0,
]


async def wait_for_turn(archive_graph, answer_id: str, turn_order: int) -> dict | None:
    """Poll the checkpointed `turns` list until the given turn_order shows up
    (or we give up and return None). Same retry tuning as the old
    tavus_controller._wait_for_archived_turns, just keyed on a single
    turn_order instead of a total expected_turn_count. Used by
    infra.message_broker.publishers.turn_publisher.publish_turn_if_new to wait
    for WPF's separate /turns/archive upload to land before publishing."""
    turn: dict | None = None
    for delay in [0.0, *_ARCHIVE_CATCHUP_RETRY_DELAYS_SECONDS]:
        if delay:
            await asyncio.sleep(delay)
        archived_state = await aget_state(archive_graph, archive_config(answer_id))
        turns = (archived_state.values or {}).get("turns") or []
        turn = next((t for t in turns if (t or {}).get("turn_order") == turn_order), None)
        if turn is not None:
            return turn

    logger.warning(
        "[archive_store] archive did not catch up before publish: answer_id=%s turn_order=%d",
        answer_id, turn_order,
    )
    return turn


async def persist_question_snapshot(
    archive_graph,
    answer_id: str,
    *,
    question,
    paper_item_id,
    language: str,
    prompt_text,
) -> None:
    """Durably snapshots the question payload for answer_id, once, right when
    AttemptConnection._handle_question_start first receives it. This is what
    lets RealtimeExamSession.create_from_archive rebuild a full session from
    nothing but answer_id on a `resume` -- the WPF client never needs to
    resend question data at all, since it's cached here the first (and only)
    time it's ever sent. Safe to call every question_start unconditionally:
    a genuinely new question just writes its own fresh values; the same
    question resumed via a fresh question_start (rather than `resume`)
    overwrites with identical values."""
    await aupdate_state(archive_graph, archive_config(answer_id), {
        "question": question,
        "paper_item_id": paper_item_id,
        "language": language,
        "prompt_text": prompt_text,
    })


async def set_current_answer_id(archive_graph, exam_attempt_id: str, answer_id: str) -> None:
    """Records answer_id as the currently-active question for exam_attempt_id, on the SAME
    archive_graph checkpointer as everything else in this module but keyed by exam_attempt_id as
    its own thread_id instead of answer_id -- a deliberate "borrow" of the per-question storage for
    an attempt-scoped concern (see FollowUpGraphState.current_answer_id's own comment), not a new
    dedicated store. Call this right at question_start, unconditionally, same as
    persist_question_snapshot -- it must land before any turn/decision exists for this question, so
    a client that lost all local state can still find out this question was started even if no
    turn ever completed for it (get_current_answer_id doesn't depend on Kafka's
    answer-turns-recorded topic, which only fires after a turn completes)."""
    await aupdate_state(archive_graph, archive_config(exam_attempt_id), {
        "current_answer_id": answer_id,
    })


async def get_current_answer_id(archive_graph, exam_attempt_id: str) -> str | None:
    """The answer_id last recorded via set_current_answer_id for this exam_attempt_id, or None if
    no question has ever been started for it (a genuinely fresh attempt)."""
    state = await aget_state(archive_graph, archive_config(exam_attempt_id))
    return (state.values or {}).get("current_answer_id")


async def persist_realtime_transcript(
    archive_graph,
    answer_id: str,
    turn_order: int,
    text: str,
    *,
    is_last_allowed_turn: bool = False,
    duration_seconds: float | None = None,
) -> None:
    """Durably saves this turn's live Voice-Live transcript (see attempt_connection.py's
    [realtime_transcript] logging) so eval-time scoring can prefer it over re-transcribing the
    archived audio via the Azure Speech SDK -- see get_realtime_transcript / start_node_config.py.
    Fire-and-forget from the caller (mirrors publish_turn_if_new's decision_reasons write):
    never awaited inline with the turn_end decision response.

    is_last_allowed_turn is persisted here too (not just used in-memory in _handle_turn_end) so
    _recover_pending_decision can re-apply WPF's own MaxTurnsPerQuestion clamp correctly if this
    turn's decision never made it back to the client and has to be recomputed during a later
    resume -- see attempt_connection.py's _handle_resume."""
    try:
        await aupdate_state(archive_graph, archive_config(answer_id), {
            "realtime_transcripts": [
                {
                    "turn_order": turn_order,
                    "text": text,
                    "is_last_allowed_turn": is_last_allowed_turn,
                    "duration_seconds": duration_seconds,
                },
            ],
        })
    except Exception:
        logger.exception(
            "[archive_store] failed to persist realtime_transcript: answer_id=%s turn_order=%d",
            answer_id, turn_order,
        )


async def get_realtime_transcript(archive_graph, answer_id: str, turn_order: int) -> str | None:
    """The live Voice-Live transcript persisted for this turn (persist_realtime_transcript),
    or None if never captured (e.g. archived data predates this feature, or the WebSocket
    dropped before turn_end for this turn)."""
    state = await aget_state(archive_graph, archive_config(answer_id))
    entries = (state.values or {}).get("realtime_transcripts") or []
    match = next((e for e in entries if e.get("turn_order") == turn_order), None)
    return match.get("text") if match else None


async def get_resume_state(archive_graph, answer_id: str) -> dict | None:
    """Full durable state for answer_id, in one Postgres round trip -- used
    by both RealtimeExamSession.hydrate_from_archive (turns + the pending
    follow-up prompt) and RealtimeExamSession.create_from_archive (the
    question snapshot, for rebuilding a session from nothing but answer_id on
    a `resume`). Returns None if nothing has ever been archived for this
    answer_id (a genuinely new question -- callers should treat that as
    "nothing to resume", not an error).

    turns are returned sorted by turn_order, with decision_reason/should_continue/next_prompt_text
    merged in from the decision_reasons marker (see publish_turn_if_new) since the archived turn
    record itself never carries them (WPF's POST /turns/archive payload has no such fields -- the
    decision is only ever known in-memory, after decide_next_step runs).

    realtime_transcripts is also returned raw (turn_order/text/is_last_allowed_turn per entry) --
    used by _recover_pending_decision to detect a turn whose transcript was captured but whose
    decision never got persisted (the connection dropped between turn_end and decide_next_step
    finishing)."""
    archived_state = await aget_state(archive_graph, archive_config(answer_id))
    values = archived_state.values or {}
    if not values.get("question") and not values.get("turns"):
        return None

    turns = list(values.get("turns") or [])
    decisions_by_turn_order = {
        int(entry.get("turn_order")): entry
        for entry in (values.get("decision_reasons") or [])
        if entry and entry.get("turn_order") is not None
    }
    for turn in turns:
        turn_order = turn.get("turn_order")
        entry = decisions_by_turn_order.get(turn_order) if turn_order is not None else None
        if entry is not None:
            turn["decision_reason"] = entry.get("reason", "")
            turn["should_continue"] = entry.get("should_continue", False)
            turn["next_prompt_text"] = entry.get("next_prompt_text")
    turns.sort(key=lambda t: t.get("turn_order") or 0)

    return {
        "turns": turns,
        "active_prompt_text": values.get("active_prompt_text"),
        "question": values.get("question"),
        "paper_item_id": values.get("paper_item_id"),
        "language": values.get("language"),
        "prompt_text": values.get("prompt_text"),
        "realtime_transcripts": list(values.get("realtime_transcripts") or []),
    }
