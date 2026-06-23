import os

from fastapi import APIRouter, Form, Request
from fastapi.encoders import jsonable_encoder

from dtos.response.follow_up import FollowUpAnswerTurn, FollowUpTurnResponse
from events import (
    AnswerTurnPayload,
    AnswerTurnsRecordedEvent,
    AnswerTurnsRecordedPayload,
)
from infra.message_broker.publishers.exam_publisher import publish_answer_turns_recorded
from infra.storage.audio_storage import download_from_s3
from node.followUpDecisionGraph.constants import MAX_TURNS
from node.state_models import QuestionContext
from utils.jsonl_logger import append_jsonl


router = APIRouter(prefix="/evaluate", tags=["FollowUp"])

FOLLOWUP_KAFKA_LOG_FILE = "followup_kafka_publish.jsonl"


def _parse_question_payload(raw_value: str | None) -> QuestionContext | None:
    if not raw_value:
        return None
    return QuestionContext.model_validate_json(raw_value)


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


@router.post("/turn")
async def evaluate_turn(
    request: Request,
    audio_ref: str = Form(...),
    answer_id: str = Form(...),
    turn_order: int = Form(...),
    prompt_text: str | None = Form(default=None),
    language: str = Form(default="en-US"),
    question: str | None = Form(default=None),
):
    if turn_order > MAX_TURNS:
        return jsonable_encoder(
            FollowUpTurnResponse(
                turn_order=turn_order,
                transcript="",
                prompt_text=prompt_text,
                current_turn=FollowUpAnswerTurn(
                    answer_id=answer_id,
                    turn_order=turn_order,
                    turn_type="MAIN" if turn_order == 1 else "FOLLOWUP",
                    prompt_text=prompt_text,
                    audio_url=audio_ref,
                    transcript="",
                    duration_seconds=None,
                    word_count=0,
                ),
                should_continue=False,
                next_prompt_text=None,
                reason="max_turns_reached",
                reached_max_turns=True,
            )
        )

    local_audio_path = download_from_s3(audio_ref)

    graph = request.app.state.followup_graph
    try:
        result = graph.invoke(
            {
                "answer_id": answer_id,
                "audio_ref": audio_ref,
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

    current_turn = result.get("current_turn") or {}
    decision = result.get("decision") or {}
    turns = result.get("turns") or []

    if not decision.get("should_continue"):
        event = AnswerTurnsRecordedEvent(
            answer_id=answer_id,
            payload=AnswerTurnsRecordedPayload(
                turns=[_build_answer_turn_payload(turn, answer_id) for turn in turns],
                reason=decision.get("reason", ""),
            ),
        )
        append_jsonl(FOLLOWUP_KAFKA_LOG_FILE, {
            "answer_id": answer_id,
            "turn_order": turn_order,
            "event": event.model_dump(by_alias=True),
        })
        await publish_answer_turns_recorded(event)

    response = FollowUpTurnResponse(
        turn_order=turn_order,
        transcript=current_turn.get("transcript", ""),
        prompt_text=current_turn.get("prompt_text"),
        current_turn=FollowUpAnswerTurn(
            answer_id=current_turn.get("answer_id"),
            turn_order=current_turn.get("turn_order", turn_order),
            turn_type=current_turn.get("turn_type"),
            prompt_text=current_turn.get("prompt_text"),
            audio_url=current_turn.get("audio_url"),
            transcript=current_turn.get("transcript", ""),
            duration_seconds=current_turn.get("duration_seconds"),
            word_count=current_turn.get("word_count"),
            answered_at=current_turn.get("answered_at"),
        ),
        should_continue=bool(decision.get("should_continue")),
        next_prompt_text=decision.get("next_prompt_text"),
        reason=decision.get("reason", ""),
        reached_max_turns=turn_order >= MAX_TURNS,
    )
    return jsonable_encoder(response)
