"""Strict validity check node — runs immediately after start_node.

Purpose: reject invalid submissions early so downstream nodes
(correction, pronunciation, LLM scoring) are never invoked.

Pure logic checks (instant, no API call):
  - audio.no_speech         : transcript is empty or word_count == 0 → reject_or_zero
  - answer_length.too_short : word_count < expected_min_words → conditional on STRICT_ZERO_ON_TOO_SHORT

LLM checks (semantic understanding):
  - safety.profanity_or_abuse      : profanity, slurs, hate, threats, severe abuse → reject_or_zero
  - task.repeats_question_only     : answer only repeats question, no original content → reject_or_zero
  - language.wrong_language_full   : transcript is mostly non-English → reject_or_zero
  - task.off_topic_full            : answer completely unrelated to question/topic → reject_or_zero

Non-blocking (scored later, not early END):
  - answer_length.too_short (when STRICT_ZERO_ON_TOO_SHORT=False) → score_with_penalty
  - fluency.excessive_fillers → handled by LLM eval nodes, not here

Only rules with blocking=true and action="reject_or_zero" route to END.
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.state_models import SpeakingInput
from node.ValidityNode.validity_eval_prompt import SYSTEM_PROMPT
from utils.length_utils import get_expected_min_words
from utils.schema_mapper import build_validity_result_from_rules, normalize_rule_result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Set to True to reject short answers with zero scores (strict exam mode).
# Set to False for development/testing — short answers get penalty but are still scored.
STRICT_ZERO_ON_TOO_SHORT = False


# ---------------------------------------------------------------------------
# Expected min words (same logic as AnswerLengthNode)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _build_llm_prompt(
    text: str,
    question_text: Optional[str],
    question_type: Optional[str],
    mode: Optional[str],
    difficulty_level: Optional[str],
    word_count: int,
) -> str:
    parts = [
        "## Transcript",
        f'"{text}"',
        "",
        "## Transcript Stats",
        f"Word count: {word_count}",
    ]

    if question_text:
        parts.append("")
        parts.append("## Question")
        parts.append(f'"{question_text}"')

    if question_type:
        parts.append(f"Question type: {question_type}")

    if mode:
        parts.append(f"Mode: {mode}")

    if difficulty_level:
        parts.append(f"Difficulty: {difficulty_level}")

    parts.append("")
    parts.append("Evaluate the transcript against all 4 rules and return the JSON result.")

    return "\n".join(parts)


def _call_llm(text: str, question_text: Optional[str], question_type: Optional[str],
              mode: Optional[str], difficulty_level: Optional[str], word_count: int) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_prompt = _build_llm_prompt(text, question_text, question_type, mode, difficulty_level, word_count)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    return json.loads(content)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


def validity_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node — strict validity gate after start_node.

    Reads: speaking_input, metadata
    Writes: validity dict into state

    Only blocking=true + action="reject_or_zero" rules route the graph to END.
    Non-blocking rules (too_short penalty, fillers) stay in rule_results for
    downstream scoring/adapter to consume.
    """

    speaking_input: Optional[SpeakingInput] = state.get("speaking_input")

    if speaking_input is None:
        return {
            **state,
            "status": "error",
            "error": "speaking_input is required for validity_node",
        }

    # Transcript source: prefer transcribed_text from start_node
    text = (
        getattr(speaking_input, "transcribed_text", None)
        or (state.get("metadata") or {}).get("transcription_text")
        or ""
    )
    text = text.strip()
    word_count = _word_count(text)

    question_text: Optional[str] = speaking_input.question.question_text if speaking_input.question else None
    question_type: Optional[str] = speaking_input.question.question_type if speaking_input.question else None
    duration_seconds: Optional[int] = speaking_input.question.duration_seconds if speaking_input.question else None
    mode: Optional[str] = getattr(speaking_input, "mode", None)
    difficulty_level: Optional[str] = speaking_input.question.difficulty_level if speaking_input.question else None

    rule_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pure logic rule 1: no_speech (always reject)
    # ------------------------------------------------------------------

    if word_count == 0:
        rule_results.append(normalize_rule_result({
            "rule_id": "audio.no_speech",
            "category": "audio",
            "severity": "critical",
            "action": "reject_or_zero",
            "blocking": True,
            "message": "No speech detected in audio.",
            "evidence": {"word_count": 0},
            "target_criteria": [],
        }))

    # ------------------------------------------------------------------
    # Pure logic rule 2: too_short (conditional)
    # ------------------------------------------------------------------

    if mode != "scripted" and word_count > 0:
        expected_min = get_expected_min_words(question_type, duration_seconds)
        if word_count < expected_min:
            if STRICT_ZERO_ON_TOO_SHORT:
                rule_results.append(normalize_rule_result({
                    "rule_id": "answer_length.too_short",
                    "category": "length",
                    "severity": "critical",
                    "action": "reject_or_zero",
                    "blocking": True,
                    "message": f"Answer has {word_count} words, expected at least {expected_min}.",
                    "evidence": {
                        "word_count": word_count,
                        "expected_min_words": expected_min,
                    },
                    "target_criteria": ["grammar", "vocabulary", "coherence"],
                }))
            else:
                # Non-blocking — scored with penalty downstream
                rule_results.append(normalize_rule_result({
                    "rule_id": "answer_length.too_short",
                    "category": "length",
                    "severity": "medium",
                    "action": "score_with_penalty",
                    "blocking": False,
                    "message": f"Answer has {word_count} words, expected at least {expected_min}. Score will be penalized.",
                    "evidence": {
                        "word_count": word_count,
                        "expected_min_words": expected_min,
                    },
                    "target_criteria": ["grammar", "vocabulary", "coherence"],
                }))

    # If pure logic already has a blocking reject, skip LLM call
    has_blocking_reject = any(
        r.get("blocking") and r.get("action") == "reject_or_zero"
        for r in rule_results
    )

    # ------------------------------------------------------------------
    # LLM rules (semantic understanding)
    # ------------------------------------------------------------------

    if not has_blocking_reject and word_count > 0:
        try:
            llm_response = _call_llm(
                text=text,
                question_text=question_text,
                question_type=question_type,
                mode=mode,
                difficulty_level=difficulty_level,
                word_count=word_count,
            )

            # Merge LLM-triggered rules
            for rule in llm_response.get("rules", []):
                if rule.get("triggered"):
                    rule_results.append(normalize_rule_result({
                        "rule_id": rule.get("rule_id", "unknown_rule"),
                        "category": str(rule.get("rule_id", "")).split(".")[0],
                        "severity": rule.get("severity", "critical"),
                        "action": rule.get("action", "reject_or_zero"),
                        "blocking": rule.get("blocking", True),
                        "message": rule.get("message", ""),
                        "evidence": rule.get("evidence", {}),
                        "target_criteria": rule.get("target_criteria", []),
                    }))

        except (json.JSONDecodeError, Exception):
            # LLM failure is non-blocking — pure logic checks have already
            # passed, so the transcript is at least minimally valid.
            pass

    # ------------------------------------------------------------------
    # Aggregate: only blocking=true + reject_or_zero routes to END
    # ------------------------------------------------------------------

    validity = build_validity_result_from_rules(
        rule_entries=rule_results,
        transcript_source="transcribed_text",
        transcript_word_count=word_count,
    )

    return {
        **state,
        "validity": validity,
        "status": "processing",
        "error": None,
    }
