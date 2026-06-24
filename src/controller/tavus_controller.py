import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from dtos.request.chat_completion import ChatCompletionRequest, ChatMessage, ToolCall
from events import AnswerTurnPayload, AnswerTurnsRecordedEvent, AnswerTurnsRecordedPayload
from infra.message_broker.publishers.exam_publisher import publish_answer_turns_recorded
from mappers.chat_completion_mapper import (
    build_tavus_message_debug_snapshot,
    build_chat_completion_chunk,
    build_chat_completion_response,
    build_end_question_tool_call,
    build_followup_state_from_messages,
    extract_answer_id,
    resolve_reply_content,
)
from utils.jsonl_logger import append_jsonl

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Tavus"])

TAVUS_CHAT_LOG_FILE = "tavus_chat.jsonl"
TAVUS_RAW_PAYLOAD_LOG_FILE = "tavus_chat_raw_payload.jsonl"
FOLLOWUP_KAFKA_LOG_FILE = "followup_kafka_publish.jsonl"


def _log_tavus_chat(stage: str, **record) -> None:
    append_jsonl(TAVUS_CHAT_LOG_FILE, {"stage": stage, **record})


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


# WPF's own /turns/archive call (S3 upload, then download-from-S3 + Azure STT) races this
# endpoint's should_end decision — Tavus's own live STT already has the transcript, so this
# call can resolve and reach here before WPF's archive for the very last turn has landed in
# Postgres. Poll briefly for the archive to catch up before publishing instead of publishing
# whatever's there immediately and silently dropping the latest turn (confirmed happening:
# question 1/3/5 all published with the last turn missing in a real run on 2026-06-24).
_ARCHIVE_CATCHUP_RETRY_DELAYS_SECONDS = [0.3, 0.3, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]


async def _wait_for_archived_turns(archive_graph, answer_id: str, expected_turn_count: int) -> list[dict]:
    turns: list[dict] = []
    for delay in [0.0, *_ARCHIVE_CATCHUP_RETRY_DELAYS_SECONDS]:
        if delay:
            await asyncio.sleep(delay)
        archived_state = archive_graph.get_state({"configurable": {"thread_id": answer_id}})
        turns = (archived_state.values or {}).get("turns") or []
        if len(turns) >= expected_turn_count:
            return turns

    logger.warning(
        "[tavus] archive did not catch up before publish: answer_id=%s expected_turns=%d got_turns=%d",
        answer_id, expected_turn_count, len(turns),
    )
    return turns


async def _publish_archived_turns(
    request: Request,
    messages: list[ChatMessage],
    decision: dict,
    expected_turn_count: int,
) -> None:
    """Publish AnswerTurnsRecordedEvent from the *archived* turns (/turns/archive
    has been persisting real S3 audio_url + Azure-quality transcripts), not the
    thin message-derived turns build_followup_state_from_messages produces."""
    answer_id = extract_answer_id(messages)
    if not answer_id:
        logger.error("[tavus] cannot publish AnswerTurnsRecordedEvent: no <answer_id> marker in messages")
        return

    archive_graph = request.app.state.archive_graph
    turns = await _wait_for_archived_turns(archive_graph, answer_id, expected_turn_count)

    event = AnswerTurnsRecordedEvent(
        answer_id=answer_id,
        payload=AnswerTurnsRecordedPayload(
            turns=[_build_answer_turn_payload(turn, answer_id) for turn in turns],
            reason=decision.get("reason", ""),
        ),
    )
    append_jsonl(FOLLOWUP_KAFKA_LOG_FILE, {
        "answer_id": answer_id,
        "event": event.model_dump(by_alias=True),
    })
    await publish_answer_turns_recorded(event)


async def _stream_chunks(model: str | None, reply_content: str, tool_calls: list[ToolCall] | None):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    content_chunk = build_chat_completion_chunk(chunk_id, created, model, content=reply_content, tool_calls=tool_calls)
    yield f"data: {content_chunk.model_dump_json()}\n\n"

    final_chunk = build_chat_completion_chunk(
        chunk_id, created, model, finish_reason="tool_calls" if tool_calls else "stop"
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    """OpenAI-compatible Custom LLM endpoint for Tavus. Public — called
    directly by Tavus, not proxied through the WPF/.NET backend.

    The only decision-maker for "does this exam question need one more
    follow-up" (docs/single-decision-source-plan.md) — reuses
    node/followUpDecisionGraph's prepare_turn_signals_node and
    followup_decision_node via the text-only build_text_followup_graph.
    When the decision is to stop: publishes AnswerTurnsRecordedEvent from
    the turns /turns/archive has archived for this answer_id, and signals
    "done" to WPF via an end_question tool call (Tavus relays tool_calls to
    the client as an app-message instead of executing them itself).
    """
    request_id = f"tavus-{uuid.uuid4().hex}"
    debug_snapshot = build_tavus_message_debug_snapshot(payload.messages)
    request_metadata = {
        "request_id": request_id,
        "model": payload.model,
        "stream": payload.stream,
        "message_count": len(payload.messages),
    }
    append_jsonl(TAVUS_RAW_PAYLOAD_LOG_FILE, {
        **request_metadata,
        "request": payload.model_dump(mode="json"),
        "debug": debug_snapshot,
    })
    _log_tavus_chat(
        "request_received",
        **request_metadata,
        debug=debug_snapshot,
    )

    try:
        state = build_followup_state_from_messages(payload.messages)
    except ValueError as exc:
        _log_tavus_chat(
            "request_invalid",
            **request_metadata,
            request=payload.model_dump(mode="json"),
            debug=debug_snapshot,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc))

    graph = request.app.state.text_followup_graph
    _log_tavus_chat(
        "followup_state_built",
        **request_metadata,
        followup_state=state,
    )

    try:
        result = graph.invoke(state)
    except Exception as exc:
        _log_tavus_chat(
            "graph_invoke_failed",
            **request_metadata,
            followup_state=state,
            error=str(exc),
        )
        raise

    if result.get("status") == "error":
        _log_tavus_chat(
            "graph_returned_error",
            **request_metadata,
            followup_state=state,
            graph_result=result,
        )
        raise HTTPException(status_code=500, detail=result.get("error") or "follow-up decision failed")

    decision = result.get("decision") or {}
    reply_content = resolve_reply_content(decision)
    should_end = not decision.get("should_continue")
    tool_calls = [build_end_question_tool_call()] if should_end else None
    _log_tavus_chat(
        "decision_resolved",
        **request_metadata,
        followup_state=state,
        graph_result=result,
        decision=decision,
        reply_content=reply_content,
        should_end=should_end,
        tool_calls=[tool_call.model_dump(mode="json") for tool_call in tool_calls or []],
    )

    if should_end:
        answer_id = extract_answer_id(payload.messages)
        archived_state = {}
        archived_turn_count = 0
        if answer_id:
            archived_state = request.app.state.archive_graph.get_state({
                "configurable": {"thread_id": answer_id},
            }).values or {}
            archived_turn_count = len(archived_state.get("turns") or [])

        _log_tavus_chat(
            "question_end_detected",
            **request_metadata,
            answer_id=answer_id,
            archived_turn_count=archived_turn_count,
            archived_state=archived_state,
        )

        try:
            await _publish_archived_turns(request, payload.messages, decision, state["turn_order"])
            _log_tavus_chat(
                "archived_turns_published",
                **request_metadata,
                answer_id=answer_id,
                archived_turn_count=archived_turn_count,
            )
        except Exception as exc:
            _log_tavus_chat(
                "archived_turns_publish_failed",
                **request_metadata,
                answer_id=answer_id,
                archived_turn_count=archived_turn_count,
                error=str(exc),
            )
            raise

    _log_tavus_chat(
        "response_ready",
        **request_metadata,
        request=payload.model_dump(mode="json"),
        debug=debug_snapshot,
        followup_state=state,
        graph_result=result,
        response={"reply_content": reply_content, "decision": decision, "stream": payload.stream},
    )

    if payload.stream:
        _log_tavus_chat(
            "streaming_response_started",
            **request_metadata,
            response={"reply_content": reply_content, "decision": decision},
        )
        return StreamingResponse(
            _stream_chunks(payload.model, reply_content, tool_calls),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = build_chat_completion_response(payload.model, reply_content, tool_calls=tool_calls)
    _log_tavus_chat(
        "non_stream_response_built",
        **request_metadata,
        response=response.model_dump(mode="json"),
    )
    return jsonable_encoder(response)
