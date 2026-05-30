"""
Controller for evaluating speaking audio using LangGraph.
"""

from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder

from node.state_models import SpeakingInput
from utils.assessment_response_adapter import adapt_current_response_to_ui_response


router = APIRouter(prefix="/evaluate", tags=["Evaluate"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _invoke_graph(
    request: Request,
    audio_path: Path,
    *,
    mode: str,
    reference_text: Optional[str] = None,
    question_id: Optional[int] = None,
    question_text: Optional[str] = None,
    question_type: Optional[str] = None,
    difficulty_level: Optional[str] = None,
    duration_seconds: Optional[int] = None,
    topic_id: Optional[int] = None,
    topic_name: Optional[str] = None,
    topic_description: Optional[str] = None,
):
    graph = request.app.state.graph

    initial_state = {
        "speaking_input": SpeakingInput(
            audio_path=str(audio_path),
            reference_text=reference_text if mode == "scripted" else None,
            mode=mode,
            language="en-US",
            question_id=question_id,
            question_text=question_text,
            question_type=question_type,
            difficulty_level=difficulty_level,
            duration_seconds=duration_seconds,
            topic_id=topic_id,
            topic_name=topic_name,
            topic_description=topic_description,
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

    old_response = {
        "status": result.get("status"),
        "error": result.get("error"),
        "audio_path": str(audio_path),
        "mode": mode,
        "reference_text": reference_text if mode == "scripted" else None,

        "question_id": question_id,
        "question_text": question_text,
        "question_type": question_type,
        "difficulty_level": difficulty_level,
        "duration_seconds": duration_seconds,

        "topic_id": topic_id,
        "topic_name": topic_name,
        "topic_description": topic_description,

        "result": result.get("pronunciation_result"),
        "metadata": {
            **result.get("metadata", {}),
            "question_id": question_id,
            "question_text": question_text,
            "question_type": question_type,
            "difficulty_level": difficulty_level,
            "duration_seconds": duration_seconds,
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_description": topic_description,
        },
        "validity": result.get("validity"),
    }

    ui_response = adapt_current_response_to_ui_response(old_response)
    return jsonable_encoder(ui_response)


# ---------------------------------------------------------------------------
# Generic endpoint — full control via query params
# ---------------------------------------------------------------------------

@router.get("/pronunciation/sample")
def evaluate_pronunciation_sample(
    request: Request,
    filename: str = Query(..., description="Audio filename inside root /data folder"),
    reference_text: Optional[str] = Query(
        default="I usually go to school by bus.",
        description="Reference sentence for scripted assessment.",
    ),
    mode: str = Query(default="scripted", description="scripted or unscripted"),
    question_id: Optional[int] = Query(default=None),
    question_text: Optional[str] = Query(default=None),
    question_type: Optional[str] = Query(default=None),
    difficulty_level: Optional[str] = Query(default=None),
    duration_seconds: Optional[int] = Query(default=None),
    topic_id: Optional[int] = Query(default=None),
    topic_name: Optional[str] = Query(default=None),
    topic_description: Optional[str] = Query(default=None),
):
    safe_filename = Path(filename).name
    audio_path = DATA_DIR / safe_filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
    if mode not in ["scripted", "unscripted"]:
        raise HTTPException(status_code=400, detail="mode must be 'scripted' or 'unscripted'")
    if mode == "scripted" and not reference_text:
        raise HTTPException(status_code=400, detail="reference_text is required when mode='scripted'")

    return _invoke_graph(
        request, audio_path,
        mode=mode, reference_text=reference_text,
        question_id=question_id, question_text=question_text,
        question_type=question_type, difficulty_level=difficulty_level,
        duration_seconds=duration_seconds, topic_id=topic_id,
        topic_name=topic_name, topic_description=topic_description,
    )


# ---------------------------------------------------------------------------
# Scenario test endpoints
# 3 audio files per scenario for comparison:
#   sample.wav       = correct pronunciation
#   sampleError.wav   = minor errors
#   sampleError2.wav  = major errors
# ---------------------------------------------------------------------------

AUDIO_FILES = [
    ("sample", "sample.wav"),
    ("sampleError", "sampleError.wav"),
    ("sampleError2", "sampleError2.wav"),
]


def _run_scenario(request: Request, *, mode: str = "unscripted", reference_text: Optional[str] = None, **kwargs):
    """Run a scenario against all 3 audio files and return comparison results."""
    results = {}
    for label, filename in AUDIO_FILES:
        audio_path = DATA_DIR / filename
        if not audio_path.exists():
            results[label] = {"error": f"File not found: {filename}"}
            continue
        results[label] = _invoke_graph(
            request, audio_path,
            mode=mode, reference_text=reference_text, **kwargs,
        )
    return results


@router.get("/test/scenario/on-topic-easy")
def test_on_topic_easy(request: Request):
    """Answer matches question perfectly. Easy, short_answer. Expect HIGH scores across all 3 audios."""
    return _run_scenario(
        request,
        question_id=1,
        question_text="How do you usually go to school?",
        question_type="short_answer",
        difficulty_level="easy",
        duration_seconds=10,
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )


@router.get("/test/scenario/off-topic")
def test_off_topic(request: Request):
    """Answer is completely off-topic. Expect LOW coherence/content scores regardless of pronunciation quality."""
    return _run_scenario(
        request,
        question_id=2,
        question_text="What is your favorite food and why do you like it?",
        question_type="long_answer",
        difficulty_level="easy",
        duration_seconds=30,
        topic_id=2,
        topic_name="Food and Cuisine",
        topic_description="Types of food, cooking methods, restaurant experiences, and dietary preferences.",
    )


@router.get("/test/scenario/too-short-for-long-answer")
def test_too_short(request: Request):
    """Answer is on-topic but too short for a description question. Expect penalized coherence/content."""
    return _run_scenario(
        request,
        question_id=3,
        question_text="Describe in detail how you commute to school every day, including what you see and experience along the way.",
        question_type="description",
        difficulty_level="medium",
        duration_seconds=60,
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )


@router.get("/test/scenario/hard-opinion")
def test_hard_opinion(request: Request):
    """Hard opinion question. Short answer expected to score lower on coherence/content."""
    return _run_scenario(
        request,
        question_id=4,
        question_text="Do you think governments should invest more in public transportation infrastructure? Why or why not?",
        question_type="opinion",
        difficulty_level="hard",
        duration_seconds=90,
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )


@router.get("/test/scenario/scripted-read-aloud")
def test_scripted(request: Request):
    """Scripted read_aloud. All LLM scores should be DIAGNOSTIC ONLY."""
    return _run_scenario(
        request,
        mode="scripted",
        reference_text="I usually go to school by bus",
        question_id=5,
        question_text="Read the following sentence aloud.",
        question_type="read_aloud",
        difficulty_level="easy",
        duration_seconds=15,
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places.",
    )


@router.get("/test/scenario/related-topic")
def test_related_topic(request: Request):
    """Answer is related to topic but doesn't directly answer the question. Expect mixed scores."""
    return _run_scenario(
        request,
        question_id=6,
        question_text="What do you think about the traffic situation in your city?",
        question_type="opinion",
        difficulty_level="medium",
        duration_seconds=45,
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )