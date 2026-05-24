"""
Controller for evaluating speaking audio using LangGraph.
"""

from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder

from node.state_models import SpeakingInput
from utils.pronunciation_formatter import format_pronunciation_api_response


router = APIRouter(prefix="/evaluate", tags=["Evaluate"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@router.get("/pronunciation/sample")
def evaluate_pronunciation_sample(
    request: Request,
    filename: str = Query(..., description="Audio filename inside root /data folder"),
    reference_text: Optional[str] = Query(
        default="I usually go to school by bus.",
        description="Reference sentence for scripted assessment.",
    ),
    mode: str = Query(default="scripted", description="scripted or unscripted"),
):
    safe_filename = Path(filename).name
    audio_path = DATA_DIR / safe_filename

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {audio_path}",
        )

    if mode not in ["scripted", "unscripted"]:
        raise HTTPException(
            status_code=400,
            detail="mode must be either 'scripted' or 'unscripted'",
        )

    if mode == "scripted" and not reference_text:
        raise HTTPException(
            status_code=400,
            detail="reference_text is required when mode='scripted'",
        )

    graph = request.app.state.graph

    initial_state = {
        "speaking_input": SpeakingInput(
            audio_path=str(audio_path),
            reference_text=reference_text if mode == "scripted" else None,
            mode=mode,
            language="en-US",
        ),
        "status": "idle",
        "metadata": {},
    }

    graph_config = {
        "configurable": {
            "thread_id": f"pronunciation-{uuid4()}",
        }
    }

    result = graph.invoke(initial_state, config=graph_config)

    pronunciation_result = result.get("pronunciation_result")

    formatted_result = None
    if pronunciation_result:
        formatted_result = format_pronunciation_api_response(
            pronunciation_result,
            mode=mode,
            reference_text=reference_text if mode == "scripted" else None,
            include_raw=False,
        )

    return jsonable_encoder(
        {
            "status": result.get("status"),
            "error": result.get("error"),
            "audio_path": str(audio_path),
            "mode": mode,
            "reference_text": reference_text if mode == "scripted" else None,
            "result": formatted_result,
            "metadata": result.get("metadata", {}),
        }
    )