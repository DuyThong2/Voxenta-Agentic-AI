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
FOLLOWUP_KAFKA_LOG_FILE = "followup_kafka_publish.jsonl"


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


async def _publish_archived_turns(request: Request, messages: list[ChatMessage], decision: dict) -> None:
    """Publish AnswerTurnsRecordedEvent from the *archived* turns (/turns/archive
    has been persisting real S3 audio_url + Azure-quality transcripts), not the
    thin message-derived turns build_followup_state_from_messages produces."""
    answer_id = extract_answer_id(messages)
    if not answer_id:
        logger.error("[tavus] cannot publish AnswerTurnsRecordedEvent: no <answer_id> marker in messages")
        return

    archive_graph = request.app.state.archive_graph
    archived_state = archive_graph.get_state({"configurable": {"thread_id": answer_id}})
    turns = (archived_state.values or {}).get("turns") or []

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

    stop_chunk = build_chat_completion_chunk(chunk_id, created, model, finish_reason="stop")
    yield f"data: {stop_chunk.model_dump_json()}\n\n"

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
    try:
        state = build_followup_state_from_messages(payload.messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    graph = request.app.state.text_followup_graph
    result = graph.invoke(state)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error") or "follow-up decision failed")

    decision = result.get("decision") or {}
    reply_content = resolve_reply_content(decision)
    should_end = not decision.get("should_continue")
    tool_calls = [build_end_question_tool_call()] if should_end else None

    if should_end:
        await _publish_archived_turns(request, payload.messages, decision)

    append_jsonl(TAVUS_CHAT_LOG_FILE, {
        "request": payload.model_dump(),
        "response": {"reply_content": reply_content, "decision": decision, "stream": payload.stream},
    })

    if payload.stream:
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
    return jsonable_encoder(response)
