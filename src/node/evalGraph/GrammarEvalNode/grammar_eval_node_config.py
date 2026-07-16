"""Grammatical range and accuracy evaluation node using LLM.

This node evaluates grammar based on the student's actual speech output
(transcribed_text), NOT the Azure reference text or scripted model answer.

Transcript priority: transcribed_text > corrected_transcript > reference_text (fallback).
See grammar_eval_node_helper for details.
"""

from typing import Any, Dict, Optional

from node.evalGraph.GrammarEvalNode.grammar_eval_node_helper import (
    build_framework_criterion_context,
    build_question_context,
    run_eval_node,
)
from node.evalGraph.GrammarEvalNode.grammar_eval_prompt import SYSTEM_PROMPT
from node.state_models import SpeakingInput
from schemas.enums import SpeakingMode


def build_user_prompt(speaking_input: SpeakingInput, transcript: str, metrics: Optional[Dict[str, Any]] = None) -> str:
    mode = speaking_input.mode or SpeakingMode.UNSCRIPTED
    question_context = build_question_context(speaking_input)
    framework_block = build_framework_criterion_context(speaking_input, "grammar")

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

    if metrics:
        parts.append("\n## Answer Length Metrics")
        parts.append(f"Word count: {metrics.get('word_count')}")
        parts.append(f"Sentence count: {metrics.get('sentence_count')}")
        parts.append(f"Length category: {metrics.get('length_category')}")
        parts.append(f"Expected min words: {metrics.get('expected_min_words')}")
        parts.append(f"Grammar range cap: {metrics.get('grammar_range_cap')}")
        parts.append(f"Code-switching ratio (non-English words / total words): {metrics.get('code_switching_ratio')}")

    if mode == SpeakingMode.SCRIPTED:
        parts.append("\nThis is a scripted read-aloud test. Grammar scores are diagnostic only.")
    else:
        parts.append("\nEvaluate grammar quality in the context of the question and difficulty level.")

    return "\n".join(parts)


def grammar_eval_node(state: dict) -> dict:
    return run_eval_node(
        state, SYSTEM_PROMPT, build_user_prompt, "grammar",
        result_key="grammar_criterion", confidence_key="grammar_confidence",
    )
