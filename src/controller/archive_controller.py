import asyncio
import os

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.encoders import jsonable_encoder

from infra.storage.audio_storage import download_from_s3_async
from node.state_models import QuestionContext

router = APIRouter(prefix="/turns", tags=["Archive"])


def _parse_question_payload(raw_value: str | None) -> QuestionContext | None:
    if not raw_value:
        return None
    return QuestionContext.model_validate_json(raw_value)


@router.post("/archive")
async def archive_turn(
    request: Request,
    audio_ref: str = Form(...),
    answer_id: str = Form(...),
    paper_item_id: str | None = Form(default=None),
    turn_order: int = Form(...),
    prompt_text: str | None = Form(default=None),
    language: str = Form(default="en-US"),
    question: str | None = Form(default=None),
):
    """Pure archival: persists this turn's S3 audio + Azure-quality transcript
    into the Postgres-checkpointed turns list keyed by answer_id, for
    /v1/chat/completions to read back and publish once the question is done.
    No decision logic here — /v1/chat/completions is the only decision-maker
    now (see docs/single-decision-source-plan.md).
    """
    local_audio_path = await download_from_s3_async(audio_ref)

    graph = request.app.state.archive_graph
    try:
        # graph.invoke (sync) not .ainvoke -- archive_graph is compiled with the
        # sync PostgresSaver checkpointer (app.py), whose aget_tuple raises
        # NotImplementedError. Running the sync call in a thread keeps this
        # route non-blocking without needing an async checkpointer.
        result = await asyncio.to_thread(
            graph.invoke,
            {
                "answer_id": answer_id,
                "audio_ref": audio_ref,
                "paper_item_id": paper_item_id,
                "question": _parse_question_payload(question),
                "language": language,
                "audio_path": local_audio_path,
                "turn_order": turn_order,
                "prompt_text": prompt_text,
                "status": "idle",
            },
            config={"configurable": {"thread_id": answer_id}},
        )
    finally:
        if local_audio_path != audio_ref and os.path.exists(local_audio_path):
            os.unlink(local_audio_path)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error") or "archive failed")

    return jsonable_encoder({"turn_order": turn_order, "status": result.get("status", "archived")})
