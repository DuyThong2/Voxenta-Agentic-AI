import json
import re
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.AnswerLengthNode.answer_length_analysis_prompt import SYSTEM_PROMPT
from node.state_models import SpeakingInput
from utils.length_utils import get_expected_min_words
from utils.transcript_selector import select_text_for_language_scoring, build_scoring_metadata


def build_user_prompt(
    speaking_input: SpeakingInput,
    transcript: str,
    word_count: int,
    sentence_count: int,
    expected_min_words: int,
    length_ratio: Optional[float],
) -> str:
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

    question_context = "\n".join(parts) if parts else "No question context provided."

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

    # Select transcript: transcribed_text > corrected_transcript > reference_text (fallback).
    transcript, _source = select_text_for_language_scoring(speaking_input)
    transcript = transcript or ""

    words = re.findall(r"\b\w+\b", transcript or "")
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", transcript) if s.strip()]
    sentence_count = len(sentences)

    question_type = speaking_input.question_type or "unknown"
    difficulty_level = speaking_input.difficulty_level or "unknown"
    duration_seconds = speaking_input.duration_seconds
    expected_min_words = get_expected_min_words(question_type, duration_seconds)

    length_ratio = word_count / expected_min_words if expected_min_words > 0 else None

    # LLM-based judgment for length_category, note, and caps
    length_category = "unknown"
    note = ""
    coherence_cap = 100
    lexical_range_cap = 100
    grammar_range_cap = 100
    llm_length_judgment = None
    length_judgment_error = None

    try:
        llm_response = call_llm_length_judgment(
            speaking_input, transcript, word_count, sentence_count,
            expected_min_words, length_ratio,
        )
        llm_length_judgment = llm_response
        length_category = llm_response.get("length_category", "unknown")
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
        "expected_min_words": expected_min_words,
        "length_ratio": round(length_ratio, 2) if length_ratio is not None else None,
        "length_category": length_category,
        "coherence_cap": coherence_cap,
        "lexical_range_cap": lexical_range_cap,
        "grammar_range_cap": grammar_range_cap,
        "note": note,
    }

    if llm_length_judgment is not None:
        metrics["llm_length_judgment"] = llm_length_judgment
    if length_judgment_error is not None:
        metrics["length_judgment_error"] = length_judgment_error

    speaking_input.answer_length_metrics = metrics

    return {
        **state,
        "speaking_input": speaking_input,
        "status": "processing",
        "metadata": {
            **state.get("metadata", {}),
            "answer_length_metrics": metrics,
        },
    }
