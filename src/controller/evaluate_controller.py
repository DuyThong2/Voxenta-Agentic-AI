import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder

from node.state_models import SpeakingInput, QuestionContext, TopicContext
from schemas.evaluation_event import EvaluationGuideInput
from schemas.enums import DifficultyLevel, QuestionType, SpeakingMode
from schemas.framework import CriterionFramework, FrameworkBand
from mappers.assessment_response_adapter import adapt_current_response_to_ui_response
from mappers.exam_event_builder import build_completed_event


router = APIRouter(prefix="/evaluate", tags=["Evaluate"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _parse_evaluation_guide_query(raw_value: Optional[str]) -> Optional[EvaluationGuideInput]:
    if not raw_value:
        return None
    return EvaluationGuideInput.model_validate_json(raw_value)


def _parse_criteria_frameworks_query(raw_value: Optional[str]) -> list[CriterionFramework]:
    if not raw_value:
        return []
    items = json.loads(raw_value)
    if not isinstance(items, list):
        raise ValueError("criteria_frameworks_json must be a JSON array")
    return [CriterionFramework.model_validate(item) for item in items]


def _coherence_framework() -> CriterionFramework:
    return CriterionFramework(
        criterion_key="coherence",
        framework_criterion_name="Coherence and Development",
        framework_criterion_description="Relevance, organization, and development of ideas.",
        bands=[
            FrameworkBand(
                code="B7",
                score_min=70,
                score_max=84,
                descriptor="Directly answers the question with clear progression and supporting detail.",
                positive_signals=["clear progression", "specific support", "direct relevance"],
                negative_signals=["minor jumps in logic"],
            ),
            FrameworkBand(
                code="B5",
                score_min=50,
                score_max=69,
                descriptor="Partly developed response with some relevant content but limited expansion.",
                positive_signals=["basic relevance"],
                negative_signals=["limited development", "loose organization"],
            ),
        ],
    )


def _grammar_framework() -> CriterionFramework:
    return CriterionFramework(
        criterion_key="grammar",
        framework_criterion_name="Grammatical Range and Accuracy",
        framework_criterion_description="Range and accurate use of grammatical structures.",
        bands=[
            FrameworkBand(
                code="B7",
                score_min=70,
                score_max=84,
                descriptor="Uses a mix of simple and complex structures with good control.",
                positive_signals=["varied tenses", "complex clauses"],
                negative_signals=["frequent basic errors"],
            ),
            FrameworkBand(
                code="B5",
                score_min=50,
                score_max=69,
                descriptor="Basic structures used accurately; complex structures attempted but with errors.",
                negative_signals=["repetitive simple structures"],
            ),
        ],
    )


def _vocabulary_framework() -> CriterionFramework:
    return CriterionFramework(
        criterion_key="vocabulary",
        framework_criterion_name="Lexical Resource",
        framework_criterion_description="Range, precision, and appropriateness of vocabulary.",
        bands=[
            FrameworkBand(
                code="B7",
                score_min=70,
                score_max=84,
                descriptor="Uses relevant vocabulary with some variety and generally accurate word choice.",
                positive_signals=["topic vocabulary", "some variety"],
                negative_signals=["occasional imprecision"],
            ),
            FrameworkBand(
                code="B5",
                score_min=50,
                score_max=69,
                descriptor="Uses mostly basic vocabulary accurately but with limited range.",
                negative_signals=["repetitive wording", "limited topic vocabulary"],
            ),
        ],
    )


def _default_criteria_frameworks() -> list[CriterionFramework]:
    return [_coherence_framework(), _grammar_framework(), _vocabulary_framework()]


def _stable_thread_id(
    *,
    exam_attempt_id: Optional[str],
    answer_id: Optional[str],
    question_id: Optional[str],
    audio_path: Path,
) -> str:
    if exam_attempt_id and answer_id:
        return f"eval-{exam_attempt_id}:{answer_id}"
    if answer_id:
        return f"eval-{answer_id}"
    if question_id:
        return f"eval-debug-{question_id}:{audio_path.stem}"
    return f"eval-debug-{audio_path.stem}"


def _invoke_graph(
    request: Request,
    audio_path: Path,
    *,
    mode: SpeakingMode,
    reference_text: Optional[str] = None,
    exam_attempt_id: Optional[str] = None,
    answer_id: Optional[str] = None,
    question_id: Optional[str] = None,
    question_text: Optional[str] = None,
    question_type: Optional[QuestionType] = None,
    difficulty_level: Optional[DifficultyLevel] = None,
    duration_seconds: Optional[int] = None,
    min_response_seconds: Optional[int] = None,
    max_response_seconds: Optional[int] = None,
    evaluation_guide: Optional[EvaluationGuideInput] = None,
    criteria_frameworks: Optional[list[CriterionFramework]] = None,
    topic_id: Optional[int] = None,
    topic_name: Optional[str] = None,
    topic_description: Optional[str] = None,
):
    graph = request.app.state.graph

    question_ctx = QuestionContext(
        question_text=question_text,
        question_type=question_type,
        difficulty_level=difficulty_level,
        duration_seconds=duration_seconds,
        min_response_seconds=min_response_seconds,
        max_response_seconds=max_response_seconds,
        evaluation_guide=evaluation_guide,
    )
    topic_ctx = TopicContext(
        topic_id=topic_id,
        topic_name=topic_name,
        topic_description=topic_description,
    )

    initial_state = {
        "speaking_input": SpeakingInput(
            exam_attempt_id=exam_attempt_id,
            answer_id=answer_id,
            question_id=question_id,
            audio_path=str(audio_path),
            reference_text=reference_text if mode == SpeakingMode.SCRIPTED else None,
            mode=mode,
            language="en-US",
            criteria_frameworks=criteria_frameworks or [],
            question=question_ctx,
            topic=topic_ctx,
        ),
        "status": "idle",
        "metadata": {},
    }

    graph_config = {
        "configurable": {
            "thread_id": _stable_thread_id(
                exam_attempt_id=exam_attempt_id,
                answer_id=answer_id,
                question_id=question_id,
                audio_path=audio_path,
            ),
        }
    }

    result = graph.invoke(initial_state, config=graph_config)

    old_response = {
        "status": result.get("status"),
        "error": result.get("error"),
        "audio_path": str(audio_path),
        "mode": mode,
        "reference_text": reference_text if mode == SpeakingMode.SCRIPTED else None,

        "question": {
            "question_id": question_id,
            "question_text": question_text,
            "question_type": question_type,
            "difficulty_level": difficulty_level,
            "duration_seconds": duration_seconds,
            "min_response_seconds": min_response_seconds,
            "max_response_seconds": max_response_seconds,
        },
        "topic": {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_description": topic_description,
        },

        "result": result.get("pronunciation_result"),
        "metadata": {
            **result.get("metadata", {}),
            "question_id": question_id,
            "question_text": question_text,
            "question_type": question_type,
            "difficulty_level": difficulty_level,
            "duration_seconds": duration_seconds,
            "min_response_seconds": min_response_seconds,
            "max_response_seconds": max_response_seconds,
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_description": topic_description,
        },
        "validity": result.get("validity"),
    }

    ui_response = adapt_current_response_to_ui_response(old_response)
    exam_event = build_completed_event(
        result,
        result.get("speaking_input"),
        audio_path=str(audio_path),
    )

    return jsonable_encoder({
        "uiResponse": ui_response,
        "examEvent": exam_event,
    })


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
    mode: SpeakingMode = Query(default=SpeakingMode.SCRIPTED, description="scripted or unscripted"),
    exam_attempt_id: Optional[str] = Query(default=None),
    answer_id: Optional[str] = Query(default=None),
    question_id: Optional[str] = Query(default=None),
    question_text: Optional[str] = Query(default=None),
    question_type: Optional[QuestionType] = Query(default=None),
    difficulty_level: Optional[DifficultyLevel] = Query(default=None),
    duration_seconds: Optional[int] = Query(default=None),
    min_response_seconds: Optional[int] = Query(default=None),
    max_response_seconds: Optional[int] = Query(default=None),
    evaluation_guide_json: Optional[str] = Query(default=None),
    criteria_frameworks_json: Optional[str] = Query(default=None),
    topic_id: Optional[int] = Query(default=None),
    topic_name: Optional[str] = Query(default=None),
    topic_description: Optional[str] = Query(default=None),
):
    safe_filename = Path(filename).name
    audio_path = DATA_DIR / safe_filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
    if mode == SpeakingMode.SCRIPTED and not reference_text:
        raise HTTPException(status_code=400, detail="reference_text is required when mode='scripted'")

    return _invoke_graph(
        request, audio_path,
        mode=mode, reference_text=reference_text,
        exam_attempt_id=exam_attempt_id,
        answer_id=answer_id,
        question_id=question_id, question_text=question_text,
        question_type=question_type, difficulty_level=difficulty_level,
        duration_seconds=duration_seconds,
        min_response_seconds=min_response_seconds,
        max_response_seconds=max_response_seconds,
        evaluation_guide=_parse_evaluation_guide_query(evaluation_guide_json),
        criteria_frameworks=_parse_criteria_frameworks_query(criteria_frameworks_json),
        topic_id=topic_id,
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


def _run_scenario(request: Request, *, mode: SpeakingMode = SpeakingMode.UNSCRIPTED, reference_text: Optional[str] = None, **kwargs):
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
        question_id="1",
        question_text="How do you usually go to school?",
        question_type=QuestionType.SHORT_ANSWER,
        difficulty_level=DifficultyLevel.EASY,
        duration_seconds=10,
        min_response_seconds=5,
        max_response_seconds=15,
        evaluation_guide=EvaluationGuideInput(
            expected_content="State how you usually travel to school.",
            key_points="transport method",
            acceptable_responses="A short direct answer is acceptable.",
            off_topic_examples="Talking only about favorite subjects.",
            scoring_hints="Do not penalize brevity if the transport method is clear.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )


@router.get("/test/scenario/off-topic")
def test_off_topic(request: Request):
    """Answer is completely off-topic. Expect LOW coherence/content scores regardless of pronunciation quality."""
    return _run_scenario(
        request,
        question_id="2",
        question_text="What is your favorite food and why do you like it?",
        question_type=QuestionType.LONG_ANSWER,
        difficulty_level=DifficultyLevel.EASY,
        duration_seconds=30,
        min_response_seconds=20,
        max_response_seconds=45,
        evaluation_guide=EvaluationGuideInput(
            expected_content="Name a favorite food and explain at least one reason.",
            key_points="food choice; reason",
            acceptable_responses="Any food is acceptable if a reason is given.",
            off_topic_examples="Discussing travel to school instead of food.",
            scoring_hints="Penalize strongly if no food preference is stated.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=2,
        topic_name="Food and Cuisine",
        topic_description="Types of food, cooking methods, restaurant experiences, and dietary preferences.",
    )


@router.get("/test/scenario/too-short-for-long-answer")
def test_too_short(request: Request):
    """Answer is on-topic but too short for a description question. Expect penalized coherence/content."""
    return _run_scenario(
        request,
        question_id="3",
        question_text="Describe in detail how you commute to school every day, including what you see and experience along the way.",
        question_type=QuestionType.DESCRIPTION,
        difficulty_level=DifficultyLevel.MEDIUM,
        duration_seconds=60,
        min_response_seconds=45,
        max_response_seconds=90,
        evaluation_guide=EvaluationGuideInput(
            expected_content="Describe the full commute with details or experiences along the way.",
            key_points="transport mode; route; observations; feelings",
            acceptable_responses="Any school commute description with concrete detail.",
            off_topic_examples="Only saying the destination without describing the journey.",
            scoring_hints="Use min response time as strong evidence for expected development.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )


@router.get("/test/scenario/hard-opinion")
def test_hard_opinion(request: Request):
    """Hard opinion question. Short answer expected to score lower on coherence/content."""
    return _run_scenario(
        request,
        question_id="4",
        question_text="Do you think governments should invest more in public transportation infrastructure? Why or why not?",
        question_type=QuestionType.OPINION,
        difficulty_level=DifficultyLevel.HARD,
        duration_seconds=90,
        min_response_seconds=45,
        max_response_seconds=120,
        evaluation_guide=EvaluationGuideInput(
            expected_content="State a clear position and support it with reasons or examples.",
            key_points="public transport investment; reasons; impact",
            acceptable_responses="Any clear stance with support.",
            off_topic_examples="Only describing personal commute without answering the policy question.",
            scoring_hints="Reward clear reasoning more than fancy vocabulary alone.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )


@router.get("/test/scenario/scripted-read-aloud")
def test_scripted(request: Request):
    """Scripted read_aloud. All LLM scores should be DIAGNOSTIC ONLY."""
    return _run_scenario(
        request,
        mode=SpeakingMode.SCRIPTED,
        reference_text="I usually go to school by bus",
        question_id="5",
        question_text="Read the following sentence aloud.",
        question_type=QuestionType.READ_ALOUD,
        difficulty_level=DifficultyLevel.EASY,
        duration_seconds=15,
        min_response_seconds=5,
        max_response_seconds=20,
        evaluation_guide=EvaluationGuideInput(
            expected_content="Read the sentence clearly and completely.",
            key_points="accurate reading; complete sentence",
            acceptable_responses="Close reading of the provided sentence.",
            off_topic_examples="Adding unrelated sentences.",
            scoring_hints="This is diagnostic only for language criteria.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places.",
    )


@router.get("/test/scenario/related-topic")
def test_related_topic(request: Request):
    """Answer is related to topic but doesn't directly answer the question. Expect mixed scores."""
    return _run_scenario(
        request,
        question_id="6",
        question_text="What do you think about the traffic situation in your city?",
        question_type=QuestionType.OPINION,
        difficulty_level=DifficultyLevel.MEDIUM,
        duration_seconds=45,
        min_response_seconds=25,
        max_response_seconds=60,
        evaluation_guide=EvaluationGuideInput(
            expected_content="Give an opinion on traffic conditions and support it with at least one reason or example.",
            key_points="opinion; traffic conditions; supporting detail",
            acceptable_responses="Any clear view on traffic if linked to the city context.",
            off_topic_examples="Only describing favorite transport without mentioning traffic.",
            scoring_hints="Distinguish related-topic answers from direct answers to the prompt.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=1,
        topic_name="Transportation",
        topic_description="Common ways people travel to work, school, or other places. Includes public transport, private vehicles, walking, and cycling.",
    )


@router.get("/test/scenario/framework-band-grammar")
def test_framework_band_grammar(request: Request):
    """Grammar scoring should see framework bands before falling back to difficulty heuristics."""
    return _run_scenario(
        request,
        question_id="7",
        question_text="Describe a place in your hometown that is important to you.",
        question_type=QuestionType.DESCRIPTION,
        difficulty_level=DifficultyLevel.MEDIUM,
        duration_seconds=60,
        min_response_seconds=35,
        max_response_seconds=75,
        criteria_frameworks=_default_criteria_frameworks(),
        evaluation_guide=EvaluationGuideInput(
            expected_content="Describe the place and explain why it matters personally.",
            key_points="location; description; personal importance",
            acceptable_responses="Any meaningful hometown place with explanation.",
            off_topic_examples="Talking only about a person without describing a place.",
            scoring_hints="Use the grammar framework as the main calibration source for grammar scoring.",
        ),
        topic_id=3,
        topic_name="Places and Hometown",
        topic_description="Descriptions of meaningful local places and personal connections to them.",
    )


@router.get("/test/scenario/evaluation-guide-content")
def test_evaluation_guide_content(request: Request):
    """Content scoring should surface evaluation-guide expectations in the shared prompt context."""
    return _run_scenario(
        request,
        question_id="8",
        question_text="Do you think working from home is better than working in an office? Why?",
        question_type=QuestionType.OPINION,
        difficulty_level=DifficultyLevel.MEDIUM,
        duration_seconds=60,
        min_response_seconds=30,
        max_response_seconds=75,
        evaluation_guide=EvaluationGuideInput(
            expected_content="An opinion (for/against/mixed) with at least one supporting reason.",
            key_points="flexibility, commute time, productivity, isolation",
            acceptable_responses="Any clear stance with a reason, even if brief.",
            off_topic_examples="Talking only about office furniture or unrelated daily routine without giving an opinion.",
            scoring_hints="Penalize heavily if no opinion is stated at all.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=4,
        topic_name="Work and Lifestyle",
        topic_description="Opinions about work habits, productivity, and modern working arrangements.",
    )


@router.get("/test/scenario/min-response-seconds")
def test_min_response_seconds(request: Request):
    """Expected minimum words should come from min_response_seconds when provided."""
    return _run_scenario(
        request,
        question_id="9",
        question_text="Describe a memorable event from your school life.",
        question_type=QuestionType.DESCRIPTION,
        difficulty_level=DifficultyLevel.MEDIUM,
        min_response_seconds=45,
        max_response_seconds=90,
        evaluation_guide=EvaluationGuideInput(
            expected_content="Describe a specific school event with enough detail to show why it was memorable.",
            key_points="event; actions; personal reflection",
            acceptable_responses="Any specific school event with some development.",
            off_topic_examples="Listing school subjects without describing an event.",
            scoring_hints="Use the minimum response time as the main anchor for expected development.",
        ),
        criteria_frameworks=_default_criteria_frameworks(),
        topic_id=5,
        topic_name="School Life",
        topic_description="Personal experiences, memories, and reflective descriptions from school.",
    )
