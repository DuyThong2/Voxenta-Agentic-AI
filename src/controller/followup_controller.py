import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder

from schemas.evaluation_event import EvaluationGuideInput
from schemas.followup import FollowUpTurnResponse


router = APIRouter(prefix="/evaluate", tags=["FollowUp"])


def _parse_guide_payload(raw_value: str | None) -> EvaluationGuideInput | None:
    if not raw_value:
        return None
    return EvaluationGuideInput.model_validate_json(raw_value)


@router.post("/turn")
async def evaluate_turn(
    request: Request,
    audio: UploadFile = File(...),
    answer_id: str = Form(...),
    turn_order: int = Form(...),
    question_text: str | None = Form(default=None),
    prompt_text: str | None = Form(default=None),
    language: str = Form(default="en-US"),
    evaluation_guide: str | None = Form(default=None),
):
    if turn_order > 3:
        return jsonable_encoder(
            FollowUpTurnResponse(
                turn_order=turn_order,
                transcript="",
                should_continue=False,
                next_prompt_text=None,
                reached_max_turns=True,
            )
        )

    suffix = Path(audio.filename or "turn.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(await audio.read())
        temp_path = tmp_file.name

    graph = request.app.state.followup_graph
    try:
        result = graph.invoke(
            {
                "answer_id": answer_id,
                "question_text": question_text,
                "evaluation_guide": _parse_guide_payload(evaluation_guide),
                "language": language,
                "audio_path": temp_path,
                "current_turn_order": turn_order,
                "current_prompt_text": prompt_text,
                "status": "idle",
            },
            config={"configurable": {"thread_id": answer_id}},
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    current_turn = result.get("current_turn") or {}
    decision = result.get("decision") or {}

    response = FollowUpTurnResponse(
        turn_order=turn_order,
        transcript=current_turn.get("transcript", ""),
        should_continue=bool(decision.get("should_continue")),
        next_prompt_text=decision.get("next_prompt_text"),
        reached_max_turns=turn_order >= 3,
    )
    return jsonable_encoder(response)
