import logging
import wave
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from node.followUpDecisionGraph.FollowUpNode.followup_decision_node_config import (
    followup_decision_node,
)
from node.followUpDecisionGraph.RepeatRecoveryNode.repeat_recovery_node_config import (
    repeat_recovery_node,
)
from node.followUpDecisionGraph.SignalNode.signal_node_config import (
    prepare_turn_signals_node,
)
from node.followUpDecisionGraph.GraphState import FollowUpGraphState
from utils.text_utils import word_count
from utils.speech_client import transcribe
from infra.message_broker import ai_usage_tracker

logger = logging.getLogger(__name__)

# transcribe() runs Azure's continuous speech recognition, which needs timely GIL access for its
# recognized/canceled callbacks. Running it in a thread (the transcribe_turn_node docstring below
# explains why this whole node stays sync) still shares the GIL with this same process's YOLO
# proctoring inference (controller/webrtc.py) -- confirmed by testing: identical audio that failed
# with zero recognized segments during a live exam (YOLO + avatar rendering running concurrently)
# transcribed correctly in isolation. A real OS process, not just a thread, is the only way to
# fully escape that GIL contention. transcribe()'s inputs (a file path, a language string) and
# output (a plain string) are trivially picklable, so this is a surgical fix -- only the
# transcription call moves to another process, not the whole archive graph/checkpointer.
_transcribe_pool = ProcessPoolExecutor(max_workers=2)


def _state_without_turns(state: FollowUpGraphState) -> Dict[str, Any]:
    """FollowUpGraphState.turns is Annotated[List, add]. Spreading **state
    back out when a node doesn't intend to touch turns still re-submits
    whatever turns already is as a "new" update, and the add reducer
    appends it again — silently duplicating history across invokes on the
    same thread_id. Strip it out of any return that isn't intentionally
    appending exactly one new turn."""
    return {k: v for k, v in state.items() if k != "turns"}


def _wav_duration_seconds(audio_path: str) -> Optional[int]:
    """WPF produces 16kHz mono PCM16 WAV files, so stdlib `wave` is enough."""
    try:
        with wave.open(audio_path, "rb") as f:
            return round(f.getnframes() / f.getframerate())
    except Exception:
        logger.warning("[transcribe_turn_node] failed to read WAV duration for %s", audio_path, exc_info=True)
        return None


def transcribe_turn_node(state: FollowUpGraphState) -> Dict[str, Any]:
    """Sync on purpose: archive_graph is compiled with the sync PostgresSaver
    checkpointer (see app.py), which only supports sync .invoke() -- LangGraph's
    sync execution path raises RuntimeError for any async def node. The
    archive_controller route stays non-blocking by running graph.invoke(...)
    itself inside asyncio.to_thread(...), not by making this node async."""
    audio_path = state.get("audio_path")
    if not audio_path:
        logger.warning(
            "[archive] thieu audio_path -- bo qua phien am answer_id=%s turn=%s",
            state.get("answer_id"), state.get("turn_order"),
        )
        return {**_state_without_turns(state), "status": "error", "error": "audio_path is required"}

    transcript = _transcribe_pool.submit(transcribe, audio_path, state.get("language", "en-US")).result() or ""

    audio_duration_seconds = state.get("duration_seconds") or _wav_duration_seconds(audio_path)

    # Log CA khi thanh cong. Moi duong that bai ben trong _transcribe_impl deu da tu ghi log
    # (Azure NoMatch / no speech segments / canceled / timed out), nhung khong co duong nao ghi
    # log khi no chay TRON VEN -- nen khi transcript ve rong ma khong co canh bao nao, khong
    # phan biet duoc "Azure tra ve rong" voi "node nay chua tung chay".
    #
    # Do chinh la be tac gap ngay 2026-08-09: turns[].transcript rong o ca checkpoint lan Kafka,
    # trong khi ca ngay khong co mot dong [transcribe] nao. Mot dong log o day tra loi dut diem.
    logger.info(
        "[archive] phien am xong answer_id=%s turn=%s chars=%d audio_path=%s",
        state.get("answer_id"), state.get("turn_order"), len(transcript), audio_path,
    )

    current_turn = {
        "answer_id": state.get("answer_id"),
        "paper_item_id": state.get("paper_item_id"),
        "turn_order": state["turn_order"],
        "turn_type": "MAIN" if state["turn_order"] == 1 else "FOLLOWUP",
        "prompt_text": state.get("prompt_text"),
        "audio_url": state.get("audio_ref"),
        "transcript": transcript,
        "word_count": word_count(transcript),
        "duration_seconds": audio_duration_seconds,
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }

    # Chi phí STT tính theo giây audio ĐÃ xử lý (đúng cơ sở tính tiền của Azure Speech), không
    # phải wall-clock thời gian gọi -- xem infra/message_broker/ai_usage_tracker.py.
    try:
        ai_usage_tracker.record_duration_usage(
            state.get("answer_id"),
            "azure_stt",
            None if audio_duration_seconds is None else audio_duration_seconds * 1000,
        )
    except Exception:
        logger.exception("[ai_usage_tracker] failed to record STT usage, ignoring")

    return {
        **_state_without_turns(state),
        "status": "processing",
        "current_turn": current_turn,
    }


def append_turn_node(state: FollowUpGraphState) -> Dict[str, Any]:
    """Append the current turn to the checkpointed turns list for this
    thread_id=answer_id. No decision logic — /v1/chat/completions is the
    only decision-maker now (docs/single-decision-source-plan.md).

    Idempotency guard: `turns` is Annotated[List, add], so a blind append
    would duplicate this turn if /turns/archive is retried (e.g. a timed-out
    request that actually succeeded server-side) for a turn_order already
    present in this thread_id's checkpointed turns. Upsert semantics instead:
    if a turn with this turn_order already exists, skip the append so the
    retried call is a safe no-op rather than a duplicate."""
    current_turn = state["current_turn"]
    existing_turns = state.get("turns") or []
    already_archived = any(
        (turn or {}).get("turn_order") == current_turn.get("turn_order")
        for turn in existing_turns
    )
    return {
        **_state_without_turns(state),
        "status": "archived",
        "turns": [] if already_archived else [current_turn],
    }


def route_on_error(state: FollowUpGraphState) -> str:
    if state.get("status") == "error":
        return "end"
    return "continue"


def merge_decision_node(state: FollowUpGraphState) -> Dict[str, Any]:
    use_repeat_recovery = bool(state.get("repeat_recovery_edge_case_handled"))
    final_decision = (
        state.get("repeat_recovery_decision")
        if use_repeat_recovery
        else state.get("followup_decision_result")
    )
    final_error = (
        state.get("repeat_recovery_error")
        if use_repeat_recovery
        else state.get("followup_decision_error")
    )

    if final_decision is None:
        final_decision = state.get("followup_decision_result") or state.get("repeat_recovery_decision") or {
            "should_continue": False,
            "next_prompt_text": None,
            "reason": "decision_fallback",
        }
        final_error = (
            final_error
            or state.get("followup_decision_error")
            or state.get("repeat_recovery_error")
            or "No decision produced by follow-up graph"
        )

    # hard_stop is enforced by code in followup_decision_node, but repeat_recovery_node only ever
    # sees it as text inside its own LLM prompt -- no code-level guarantee there. If
    # repeat_recovery_node's decision wins the merge above (use_repeat_recovery=True) while
    # hard_stop is set, its LLM-chosen should_continue=True would otherwise slip through
    # ungoverned. Enforce hard_stop here, once, as the single point every path converges on,
    # regardless of which node produced final_decision.
    signals = state.get("signals") or {}
    if signals.get("hard_stop") and final_decision.get("should_continue"):
        final_decision = {
            "should_continue": False,
            "next_prompt_text": None,
            "reason": signals.get("hard_stop_reason") or "max_turns_reached",
        }

    return {
        "decision": final_decision,
        "status": "completed",
        "error": final_error,
    }


def build_archive_graph(checkpointer):
    """Archive-only graph: transcribe the turn (reusing transcribe_turn_node
    as-is) and append it to the Postgres-checkpointed turns list keyed by
    thread_id=answer_id, for /v1/chat/completions to read back and publish
    once the question is done."""
    g = StateGraph(FollowUpGraphState)
    g.add_node("transcribe_turn", transcribe_turn_node)
    g.add_node("append_turn", append_turn_node)

    g.add_edge(START, "transcribe_turn")
    g.add_conditional_edges(
        "transcribe_turn",
        route_on_error,
        {
            "end": END,
            "continue": "append_turn",
        },
    )
    g.add_edge("append_turn", END)

    return g.compile(checkpointer=checkpointer)


def build_text_followup_graph():
    """Text-only variant of the follow-up graph: skips transcribe_turn since
    the caller (Tavus's /v1/chat/completions) already sends text, not audio.

    No checkpointer: the caller resends the full message history every call,
    so each invocation is stateless.
    """
    g = StateGraph(FollowUpGraphState)
    g.add_node("prepare_turn_signals", prepare_turn_signals_node)
    g.add_node("repeat_recovery", repeat_recovery_node)
    g.add_node("followup_decision", followup_decision_node)
    g.add_node("merge_decision", merge_decision_node)

    g.add_edge(START, "prepare_turn_signals")
    g.add_edge("prepare_turn_signals", "repeat_recovery")
    g.add_edge("prepare_turn_signals", "followup_decision")
    g.add_edge("repeat_recovery", "merge_decision")
    g.add_edge("followup_decision", "merge_decision")
    g.add_edge("merge_decision", END)

    return g.compile()
