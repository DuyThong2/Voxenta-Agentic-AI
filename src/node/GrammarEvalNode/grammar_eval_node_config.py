"""Grammatical range and accuracy evaluation node using LLM.

This node evaluates grammar based on the student's actual speech output
(transcribed_text), NOT the Azure reference text or scripted model answer.

Transcript priority: transcribed_text > corrected_transcript > reference_text (fallback).
See utils.transcript_selector for details.
"""

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.GrammarEvalNode.grammar_eval_prompt import SYSTEM_PROMPT
from node.state_models import SpeakingInput
from utils.transcript_selector import select_text_for_language_scoring, build_scoring_metadata
from utils.question_context_helper import build_question_context


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
        parts.append(f"Grammar range cap: {speaking_input.answer_length_metrics.get('grammar_range_cap')}")

    if mode == "scripted":
        parts.append("\nThis is a scripted read-aloud test. Grammar scores are diagnostic only.")
    else:
        parts.append("\nEvaluate grammar quality in the context of the question and difficulty level.")

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


def merge_grammar_score(pronunciation_result: Any, llm_response: Dict[str, Any]) -> Any:
    """Merge grammatical range and accuracy score into pronunciation_result.criteria."""
    criteria = dict(pronunciation_result.criteria)

    criteria["grammatical_range_and_accuracy"] = {
        "score": llm_response["score"],
        "subscores": llm_response.get("subscores", {}),
        "note": llm_response.get("note", "Evaluated by LLM based on transcript analysis."),
    }

    pronunciation_result.criteria = criteria
    return pronunciation_result


def grammar_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to evaluate grammatical range and accuracy using LLM.

    Reads: speaking_input, pronunciation_result
    Updates: pronunciation_result.criteria.grammatical_range_and_accuracy
    """

    speaking_input = state.get("speaking_input")
    pronunciation_result = state.get("pronunciation_result")

    if speaking_input is None:
        return {**state, "status": "error", "error": "speaking_input is required for grammar_eval_node"}

    if pronunciation_result is None:
        return {**state, "status": "error", "error": "pronunciation_result is required. Run pronunciation_eval first."}

    # Select transcript: transcribed_text > corrected_transcript > reference_text (fallback).
    # Do NOT use Azure reference_text as the primary scoring input.
    transcript, source = select_text_for_language_scoring(speaking_input)

    if transcript is None:
        return {**state, "status": "error", "error": "No transcript available for grammar evaluation"}

    scoring_meta = build_scoring_metadata(source, speaking_input.mode)

    try:
        llm_response = call_llm(speaking_input, transcript)
        updated_result = merge_grammar_score(pronunciation_result, llm_response)

        # Attach transcript source metadata so downstream knows which text was scored.
        existing_meta = state.get("metadata") or {}
        merged_meta = {**existing_meta, **scoring_meta}

        return {**state, "pronunciation_result": updated_result, "metadata": merged_meta, "status": "completed", "error": None}

    except json.JSONDecodeError as exc:
        return {**state, "status": "error", "error": f"LLM returned invalid JSON: {str(exc)}"}

    except Exception as exc:
        return {**state, "status": "error", "error": f"Grammar evaluation failed: {str(exc)}"}
