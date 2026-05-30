"""Coherence evaluation node using LLM to assess logical flow and connected discourse.

This node evaluates coherence based on the student's actual speech output
(transcribed_text), NOT the Azure reference text or scripted model answer.

Transcript priority: transcribed_text > corrected_transcript > reference_text (fallback).
See utils.transcript_selector for details.
"""

from node.CoherenceEvalNode.coherence_eval_prompt import SYSTEM_PROMPT
from node.state_models import SpeakingInput
from utils.eval_node_helper import run_eval_node
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
        parts.append(f"Coherence cap: {speaking_input.answer_length_metrics.get('coherence_cap')}")

    if mode == "scripted":
        parts.append("\nThis is a scripted read-aloud test. Coherence scores are diagnostic only.")
    else:
        parts.append("\nEvaluate whether the answer is relevant to the question and topic, then assess coherence.")

    return "\n".join(parts)


def coherence_eval_node(state: dict) -> dict:
    return run_eval_node(state, "coherence", SYSTEM_PROMPT, build_user_prompt, "coherence")
