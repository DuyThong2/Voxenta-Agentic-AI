"""Coherence evaluation node using LLM to assess logical flow and connected discourse."""

import json
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.CoherenceEvalNode.coherence_eval_prompt import SYSTEM_PROMPT
from node.state_models import SpeakingInput


def build_question_context(speaking_input: SpeakingInput) -> str:
    """Build question/topic context block for the LLM prompt."""
    parts = []

    if speaking_input.question_text:
        parts.append(f'Question: "{speaking_input.question_text}"')

    if speaking_input.question_type:
        parts.append(f"Question type: {speaking_input.question_type}")

    if speaking_input.difficulty_level:
        parts.append(f"Difficulty: {speaking_input.difficulty_level}")

    if speaking_input.duration_seconds is not None:
        parts.append(f"Expected duration: {speaking_input.duration_seconds}s")

    if speaking_input.topic_name:
        parts.append(f"Topic: {speaking_input.topic_name}")

    if speaking_input.topic_description:
        parts.append(f"Topic description: {speaking_input.topic_description}")

    return "\n".join(parts) if parts else "No question context provided."


def build_user_prompt(speaking_input: SpeakingInput, transcript: str, reference_text: Optional[str]) -> str:
    mode = speaking_input.mode or "unscripted"
    question_context = build_question_context(speaking_input)

    parts = [
        "## Question Context",
        question_context,
        "",
        "## Speaker's Answer",
        f"Mode: {mode}",
        f'Transcript: "{transcript}"',
    ]

    if reference_text:
        parts.append(f'Reference text: "{reference_text}"')

    if speaking_input.answer_length_metrics:
        parts.append("\n## Answer Length Metrics")
        parts.append(f"Word count: {speaking_input.answer_length_metrics.get('word_count')}")
        parts.append(f"Sentence count: {speaking_input.answer_length_metrics.get('sentence_count')}")
        parts.append(f"Length category: {speaking_input.answer_length_metrics.get('length_category')}")
        parts.append(f"Expected min words: {speaking_input.answer_length_metrics.get('expected_min_words')}")
        parts.append(f"Coherence cap: {speaking_input.answer_length_metrics.get('coherence_cap')}")

    if mode == "scripted":
        parts.append("\nThis is a scripted read-aloud test. Coherence scores are diagnostic only.")
    else:
        parts.append("\nEvaluate whether the answer is relevant to the question and topic, then assess coherence.")

    return "\n".join(parts)


def call_llm(speaking_input: SpeakingInput, transcript: str, reference_text: Optional[str]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(speaking_input, transcript, reference_text)),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    return json.loads(content)


def merge_coherence_score(pronunciation_result: Any, llm_response: Dict[str, Any]) -> Any:
    """Merge coherence score into pronunciation_result.criteria.fluency_and_coherence."""
    criteria = dict(pronunciation_result.criteria)

    fc = dict(criteria.get("fluency_and_coherence", {}))
    subscores = dict(fc.get("subscores", {}))

    coherence_score = llm_response["score"]
    subscores["coherence"] = coherence_score

    fluency_score = subscores.get("fluency")
    if fluency_score is not None:
        fc["score"] = round((fluency_score + coherence_score) / 2, 1)

    fc["subscores"] = subscores
    fc["note"] = llm_response.get("note", "Fluency from audio. Coherence from LLM analysis.")
    criteria["fluency_and_coherence"] = fc

    pronunciation_result.criteria = criteria
    return pronunciation_result


def coherence_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to evaluate coherence using LLM.

    Reads: speaking_input, pronunciation_result
    Updates: pronunciation_result.criteria.fluency_and_coherence.coherence
    """

    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")

    if speaking_input is None:
        return {**state, "status": "error", "error": "speaking_input is required for coherence_eval_node"}

    if pronunciation_result is None:
        return {**state, "status": "error", "error": "pronunciation_result is required. Run pronunciation_eval first."}

    transcript = (
        speaking_input.corrected_transcript
        or speaking_input.transcribed_text
        or pronunciation_result.recognized_text
    )

    if not transcript:
        return {**state, "status": "error", "error": "No transcript available for coherence evaluation"}

    reference_text = speaking_input.reference_text

    try:
        llm_response = call_llm(speaking_input, transcript, reference_text)
        updated_result = merge_coherence_score(pronunciation_result, llm_response)

        return {**state, "pronunciation_result": updated_result, "status": "processing", "error": None}

    except json.JSONDecodeError as exc:
        return {**state, "status": "error", "error": f"LLM returned invalid JSON: {str(exc)}"}

    except Exception as exc:
        return {**state, "status": "error", "error": f"Coherence evaluation failed: {str(exc)}"}
