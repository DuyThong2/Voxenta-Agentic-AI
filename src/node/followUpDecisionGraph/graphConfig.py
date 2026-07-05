import logging
import wave
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

logger = logging.getLogger(__name__)


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
    audio_path = state.get("audio_path")
    if not audio_path:
        return {**_state_without_turns(state), "status": "error", "error": "audio_path is required"}

    transcript = transcribe(audio_path, state.get("language", "en-US")) or ""

    current_turn = {
        "answer_id": state.get("answer_id"),
        "turn_order": state["turn_order"],
        "turn_type": "MAIN" if state["turn_order"] == 1 else "FOLLOWUP",
        "prompt_text": state.get("prompt_text"),
        "audio_url": state.get("audio_ref"),
        "transcript": transcript,
        "word_count": word_count(transcript),
        "duration_seconds": _wav_duration_seconds(audio_path),
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }

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


def route_after_repeat_recovery(state: FollowUpGraphState) -> str:
    if state.get("status") == "error":
        return "end"
    if state.get("edge_case_handled"):
        return "end"
    return "continue"


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

    g.add_edge(START, "prepare_turn_signals")
    g.add_edge("prepare_turn_signals", "repeat_recovery")
    g.add_conditional_edges(
        "repeat_recovery",
        route_after_repeat_recovery,
        {
            "end": END,
            "continue": "followup_decision",
        },
    )
    g.add_edge("followup_decision", END)

    return g.compile()
