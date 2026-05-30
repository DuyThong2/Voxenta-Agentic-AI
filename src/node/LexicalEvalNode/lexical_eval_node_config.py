"""Lexical resource evaluation node using LLM to assess vocabulary range and accuracy.

This node evaluates vocabulary based on the student's actual speech output
(transcribed_text), NOT the Azure reference text or scripted model answer.

Transcript priority: transcribed_text > corrected_transcript > reference_text (fallback).
See utils.transcript_selector for details.
"""

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.LexicalEvalNode.lexical_eval_prompt import SYSTEM_PROMPT
from node.state_models import SpeakingInput
from utils.transcript_selector import select_text_for_language_scoring, build_scoring_metadata


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


def build_user_prompt(speaking_input: SpeakingInput, transcript: str) -> str:
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

    if speaking_input.answer_length_metrics:
        parts.append("\n## Answer Length Metrics")
        parts.append(f"Word count: {speaking_input.answer_length_metrics.get('word_count')}")
        parts.append(f"Sentence count: {speaking_input.answer_length_metrics.get('sentence_count')}")
        parts.append(f"Length category: {speaking_input.answer_length_metrics.get('length_category')}")
        parts.append(f"Expected min words: {speaking_input.answer_length_metrics.get('expected_min_words')}")
        parts.append(f"Lexical range cap: {speaking_input.answer_length_metrics.get('lexical_range_cap')}")

    if mode == "scripted":
        parts.append("\nThis is a scripted read-aloud test. Vocabulary scores are diagnostic only.")
    else:
        parts.append("\nEvaluate whether the vocabulary is relevant to the question and topic, then assess range and accuracy.")

    return "\n".join(parts)


def call_llm(speaking_input: SpeakingInput, transcript: str) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(speaking_input, transcript)),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    return json.loads(content)


def merge_lexical_score(pronunciation_result: Any, llm_response: Dict[str, Any]) -> Any:
    """Merge lexical resource score into pronunciation_result.criteria."""
    criteria = dict(pronunciation_result.criteria)

    criteria["lexical_resource"] = {
        "score": llm_response["score"],
        "subscores": llm_response.get("subscores", {}),
        "note": llm_response.get("note", "Evaluated by LLM based on transcript analysis."),
    }

    pronunciation_result.criteria = criteria
    return pronunciation_result


def lexical_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to evaluate lexical resource using LLM.

    Reads: speaking_input, pronunciation_result
    Updates: pronunciation_result.criteria.lexical_resource
    """

    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")

    if speaking_input is None:
        return {**state, "status": "error", "error": "speaking_input is required for lexical_eval_node"}

    if pronunciation_result is None:
        return {**state, "status": "error", "error": "pronunciation_result is required. Run pronunciation_eval first."}

    # Select transcript: transcribed_text > corrected_transcript > reference_text (fallback).
    # Do NOT use Azure reference_text as the primary scoring input.
    transcript, source = select_text_for_language_scoring(speaking_input)

    if transcript is None:
        return {**state, "status": "error", "error": "No transcript available for lexical evaluation"}

    scoring_meta = build_scoring_metadata(source, speaking_input.mode)

    try:
        llm_response = call_llm(speaking_input, transcript)
        updated_result = merge_lexical_score(pronunciation_result, llm_response)

        existing_meta = state.get("metadata") or {}
        merged_meta = {**existing_meta, **scoring_meta}

        return {**state, "pronunciation_result": updated_result, "metadata": merged_meta, "status": "completed", "error": None}

    except json.JSONDecodeError as exc:
        return {**state, "status": "error", "error": f"LLM returned invalid JSON: {str(exc)}"}

    except Exception as exc:
        return {**state, "status": "error", "error": f"Lexical evaluation failed: {str(exc)}"}
