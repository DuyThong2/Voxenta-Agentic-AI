import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from dtos.request.chat_completion import ChatCompletionRequest
from mappers.chat_completion_mapper import (
    build_chat_completion_chunk,
    build_chat_completion_response,
    build_followup_state_from_messages,
    resolve_reply_content,
)
from utils.jsonl_logger import append_jsonl

router = APIRouter(prefix="/v1", tags=["Tavus"])

TAVUS_CHAT_LOG_FILE = "tavus_chat.jsonl"


async def _stream_chunks(model: str | None, reply_content: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    content_chunk = build_chat_completion_chunk(chunk_id, created, model, content=reply_content)
    yield f"data: {content_chunk.model_dump_json()}\n\n"

    stop_chunk = build_chat_completion_chunk(chunk_id, created, model, finish_reason="stop")
    yield f"data: {stop_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    """OpenAI-compatible Custom LLM endpoint for Tavus. Public — called
    directly by Tavus, not proxied through the WPF/.NET backend.

    Reuses node/followUpDecisionGraph's prepare_turn_signals_node and
    followup_decision_node (via the text-only build_text_followup_graph)
    to decide whether one more follow-up is needed.
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

    append_jsonl(TAVUS_CHAT_LOG_FILE, {
        "request": payload.model_dump(),
        "response": {"reply_content": reply_content, "decision": decision, "stream": payload.stream},
    })

    if payload.stream:
        return StreamingResponse(
            _stream_chunks(payload.model, reply_content),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = build_chat_completion_response(payload.model, reply_content)
    return jsonable_encoder(response)
