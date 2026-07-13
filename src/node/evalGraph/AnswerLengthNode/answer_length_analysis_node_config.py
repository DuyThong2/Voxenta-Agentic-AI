import json
import logging
import os
import re
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.evalGraph.AnswerLengthNode.answer_length_analysis_prompt import SYSTEM_PROMPT
from node.evalGraph.AnswerLengthNode.answer_length_node_helper import (
    build_question_context,
    select_text_for_language_scoring,
)
from node.state_models import SpeakingInput
from utils.length_utils import get_expected_min_words
from utils.speech_client import extract_non_target_segments, strip_non_target_segments

logger = logging.getLogger(__name__)


ENFORCE_CAP_IN_PYTHON = os.getenv("ENFORCE_CAP_IN_PYTHON", "true").lower() == "true"


def build_user_prompt(
    speaking_input: SpeakingInput,
    transcript: str,
    word_count: int,
    sentence_count: int,
    expected_min_words: int,
    length_ratio: Optional[float],
) -> str:
    question_context = build_question_context(speaking_input)

    return (
        "## Question Context\n"
        f"{question_context}\n"
        "\n## Speaker's Answer\n"
        f'Transcript: "{transcript}"\n'
        "\n## Computed Metrics\n"
        f"Word count: {word_count}\n"
        f"Sentence count: {sentence_count}\n"
        f"Expected min words: {expected_min_words}\n"
        f"Length ratio: {round(length_ratio, 2) if length_ratio is not None else 'N/A'}\n"
        "\nEvaluate the answer length and development based on the question context and transcript."
    )


def call_llm_length_judgment(
    speaking_input: SpeakingInput,
    transcript: str,
    word_count: int,
    sentence_count: int,
    expected_min_words: int,
    length_ratio: Optional[float],
) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(
            speaking_input, transcript, word_count, sentence_count,
            expected_min_words, length_ratio,
        )),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    return json.loads(content)


def answer_length_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    speaking_input = state.get("speaking_input")

    if speaking_input is None:
        return {**state, "status": "error", "error": "speaking_input is required for answer_length_analysis_node"}

    answer_id = getattr(speaking_input, "answer_id", None)
    turn_order = (state.get("metadata") or {}).get("turn_order")
    logger.info("[eval:answer_length] analyzing answer_id=%s turn=%s", answer_id, turn_order)

    # Select transcript: transcribed_text > corrected_transcript > reference_text (fallback).
    transcript, _source = select_text_for_language_scoring(speaking_input)
    transcript = transcript or ""

    # Code-switched segments (wrapped "[XX: ...]" by speech_client.transcribe) are excluded
    # from word/sentence counts -- only target-language words feed the length signal.
    target_language_text = strip_non_target_segments(transcript)

    words = re.findall(r"\b\w+\b", target_language_text)
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", target_language_text) if s.strip()]
    sentence_count = len(sentences)

    non_target_word_count = sum(
        len(re.findall(r"\b\w+\b", segment))
        for segment in extract_non_target_segments(transcript)
    )
    total_words_all_languages = word_count + non_target_word_count
    code_switching_ratio = (
        round(non_target_word_count / total_words_all_languages, 2)
        if total_words_all_languages > 0
        else 0.0
    )

    question_type = (speaking_input.question.question_type if speaking_input.question else None) or "unknown"
    difficulty_level = (speaking_input.question.difficulty_level if speaking_input.question else None) or "unknown"
    duration_seconds = speaking_input.question.duration_seconds if speaking_input.question else None
    min_response_seconds = speaking_input.question.min_response_seconds if speaking_input.question else None
    max_response_seconds = speaking_input.question.max_response_seconds if speaking_input.question else None
    expected_min_words = get_expected_min_words(
        question_type,
        duration_seconds,
        min_response_seconds=min_response_seconds,
    )

    length_ratio = word_count / expected_min_words if expected_min_words > 0 else None

    if length_ratio is None:
        length_category = "unknown"
    elif length_ratio < 0.5:
        length_category = "too_short"
    elif length_ratio < 0.9:
        length_category = "somewhat_short"
    elif length_ratio <= 1.8:
        length_category = "appropriate"
    else:
        length_category = "too_long_or_verbose"

    note = ""
    coherence_cap = None
    lexical_range_cap = None
    grammar_range_cap = None
    llm_length_judgment = None
    length_judgment_error = None

    if ENFORCE_CAP_IN_PYTHON:
        try:
            llm_response = call_llm_length_judgment(
                speaking_input, transcript, word_count, sentence_count,
                expected_min_words, length_ratio,
            )
            llm_length_judgment = llm_response
            length_category = llm_response.get("length_category", length_category)
            note = llm_response.get("note", "")
            coherence_cap = llm_response.get("coherence_cap", 100)
            lexical_range_cap = llm_response.get("lexical_range_cap", 100)
            grammar_range_cap = llm_response.get("grammar_range_cap", 100)
        except json.JSONDecodeError as exc:
            length_judgment_error = f"LLM returned invalid JSON: {str(exc)}"
        except Exception as exc:
            length_judgment_error = str(exc)

    metrics = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "question_type": question_type,
        "difficulty_level": difficulty_level,
        "duration_seconds": duration_seconds,
        "min_response_seconds": min_response_seconds,
        "max_response_seconds": max_response_seconds,
        "expected_min_words": expected_min_words,
        "length_ratio": round(length_ratio, 2) if length_ratio is not None else None,
        "length_category": length_category,
        "note": note,
        "caps_enforced_in_python": ENFORCE_CAP_IN_PYTHON,
        "code_switching_ratio": code_switching_ratio,
        "asr_confidence_avg": speaking_input.asr_confidence,
    }

    if coherence_cap is not None:
        metrics["coherence_cap"] = coherence_cap
    if lexical_range_cap is not None:
        metrics["lexical_range_cap"] = lexical_range_cap
    if grammar_range_cap is not None:
        metrics["grammar_range_cap"] = grammar_range_cap

    if llm_length_judgment is not None:
        metrics["llm_length_judgment"] = llm_length_judgment
    if length_judgment_error is not None:
        metrics["length_judgment_error"] = length_judgment_error

    speaking_input.answer_length_metrics = metrics

    logger.info(
        "[eval:answer_length] done answer_id=%s turn=%s word_count=%d category=%s",
        answer_id, turn_order, word_count, length_category,
    )

    return {
        **state,
        "speaking_input": speaking_input,
        "status": "processing",
        "metadata": {
            **state.get("metadata", {}),
            "answer_length_metrics": metrics,
        },
    }
