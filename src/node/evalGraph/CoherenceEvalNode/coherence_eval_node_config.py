"""Coherence evaluation node using LLM to assess logical flow and connected discourse.

This node evaluates coherence based on the student's actual speech output
(transcribed_text), NOT the Azure reference text or scripted model answer.

Transcript priority: transcribed_text > corrected_transcript > reference_text (fallback).
See coherence_eval_node_helper for details.
"""

from node.evalGraph.CoherenceEvalNode.coherence_eval_node_helper import (
    build_framework_criterion_context,
    build_question_context,
    run_eval_node,
)
from node.evalGraph.CoherenceEvalNode.coherence_eval_prompt import SYSTEM_PROMPT
from node.state_models import SpeakingInput
from schemas.enums import SpeakingMode


def build_user_prompt(speaking_input: SpeakingInput, transcript: str) -> str:
    mode = speaking_input.mode or SpeakingMode.UNSCRIPTED
    question_context = build_question_context(speaking_input)
    framework_block = build_framework_criterion_context(speaking_input, "coherence")

    parts = [
        "## Question Context",
        question_context,
    ]

    if framework_block:
        parts.append("")
        parts.append(framework_block)

    parts.extend(
        [
            "",
            "## Speaker's Answer",
            f"Mode: {mode}",
            f'Transcript: "{transcript}"',
        ]
    )

    if speaking_input.conversation_transcript:
        parts.append("")
        parts.append("## Conversation Context (for reference only, do not grade the AI's lines)")
        parts.append(
            "The lines below show the full AI/User dialogue with timestamps, including any "
            "follow-up questions. Use it only to judge whether the speaker's answer coherently "
            "follows what was actually asked -- never grade or quote the \"AI:\" lines as if "
            "they were the speaker's own words."
        )
        parts.append(speaking_input.conversation_transcript)

    if speaking_input.answer_length_metrics:
        parts.append("\n## Answer Length Metrics")
        parts.append(f"Word count: {speaking_input.answer_length_metrics.get('word_count')}")
        parts.append(f"Sentence count: {speaking_input.answer_length_metrics.get('sentence_count')}")
        parts.append(f"Length category: {speaking_input.answer_length_metrics.get('length_category')}")
        parts.append(f"Expected min words: {speaking_input.answer_length_metrics.get('expected_min_words')}")
        parts.append(f"Coherence cap: {speaking_input.answer_length_metrics.get('coherence_cap')}")

    if mode == SpeakingMode.SCRIPTED:
        parts.append("\nThis is a scripted read-aloud test. Coherence scores are diagnostic only.")
    else:
        parts.append("\nEvaluate whether the answer is relevant to the question and topic, then assess coherence.")

    return "\n".join(parts)


def coherence_eval_node(state: dict) -> dict:
    return run_eval_node(state, "coherence", SYSTEM_PROMPT, build_user_prompt, "coherence")
