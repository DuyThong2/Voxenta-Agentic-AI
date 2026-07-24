"""Strict validity check node — runs immediately after start_node.

Purpose: reject invalid submissions early so downstream nodes
(pronunciation, LLM scoring) are never invoked.

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
import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from node.state_models import SpeakingInput
from node.evalGraph.ValidityNode.validity_eval_prompt import SYSTEM_PROMPT
from schemas.enums import DifficultyLevel, SpeakingMode
from mappers.schema_mapper import build_validity_result_from_rules, normalize_rule_result
from utils.length_utils import get_expected_min_words
from utils.speech_client import compute_code_switching_ratio
from utils.text_utils import word_count

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Set to True to reject short answers with zero scores (strict exam mode).
# Set to False for development/testing — short answers get penalty but are still scored.
STRICT_ZERO_ON_TOO_SHORT = False

# Deterministic backstop for language.wrong_language_full — matches the ">70% non-English"
# threshold already stated in validity_eval_prompt.py's LLM rule, computed the SAME way
# AnswerLengthNode computes codeSwitchingRatio (speech_client.compute_code_switching_ratio,
# based on Azure's own "[XX: ...]" language-tagged segments). This does NOT replace the LLM
# rule — Azure only tags EN/VI, so a genuinely different third language or gibberish still
# needs the LLM's semantic judgment — but it removes the single-LLM-call as the ONLY thing
# standing between an unambiguously non-English answer and a passing grade.
WRONG_LANGUAGE_RATIO_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Expected min words (same logic as AnswerLengthNode)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rule_attr(rule: Any, key: str) -> Any:
    if isinstance(rule, dict):
        return rule.get(key)
    return getattr(rule, key, None)


def _coerce_positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _round_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 2)


def _build_llm_prompt(
    text: str,
    question_text: Optional[str],
    asset_context: Optional[str],
    question_type: Optional[str],
    mode: Optional[str],
    difficulty_level: Optional[str],
    word_count: int,
    actual_response_seconds: Optional[float] = None,
    expected_min_response_seconds: Optional[float] = None,
    off_topic_examples: Optional[str] = None,
) -> str:
    parts = [
        "## Transcript",
        f'"{text}"',
        "",
        "## Transcript Stats",
        f"Word count: {word_count}",
    ]
    if actual_response_seconds is not None:
        parts.append(f"Actual response duration: {round(actual_response_seconds, 1)}s")
    if expected_min_response_seconds is not None:
        parts.append(f"Expected minimum response duration: {round(expected_min_response_seconds, 1)}s")

    if question_text:
        parts.append("")
        parts.append("## Question")
        parts.append(f'"{question_text}"')

    if asset_context:
        parts.append("")
        parts.append("## Question Asset")
        parts.append(asset_context)

    if question_type:
        parts.append(f"Question type: {question_type}")

    if mode:
        parts.append(f"Mode: {mode}")

    if difficulty_level:
        parts.append(f"Difficulty: {difficulty_level}")

    if off_topic_examples:
        parts.append("")
        parts.append("## Off-topic Examples (supporting reference, not exhaustive)")
        parts.append(off_topic_examples)

    parts.append("")
    parts.append("Evaluate the transcript against all 4 rules and return the JSON result.")

    return "\n".join(parts)


def _call_llm(text: str, question_text: Optional[str], question_type: Optional[str],
              asset_context: Optional[str], mode: Optional[str], difficulty_level: Optional[str], word_count: int,
              actual_response_seconds: Optional[float] = None,
              expected_min_response_seconds: Optional[float] = None,
              off_topic_examples: Optional[str] = None) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_prompt = _build_llm_prompt(
        text, question_text, asset_context, question_type, mode, difficulty_level, word_count,
        actual_response_seconds=actual_response_seconds,
        expected_min_response_seconds=expected_min_response_seconds,
        off_topic_examples=off_topic_examples,
    )

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

    answer_id = getattr(speaking_input, "answer_id", None)
    metadata = state.get("metadata") or {}
    turn_order = metadata.get("turn_order")
    logger.info("[eval:validity] checking answer_id=%s turn=%s", answer_id, turn_order)

    # Transcript source: transcribed_text from start_node (Azure's own transcription,
    # already auto-detect + language-tagged for code-switched Vietnamese -- see
    # speech_client.transcribe()). conversation_transcript is deliberately NOT used here --
    # it's the AI/User dialogue scaffold meant only as extra context for CoherenceEvalNode
    # (see select_text_for_language_scoring's own docstring in the eval-node helpers);
    # validity must judge the student's own words, same as grammar/lexical/answer-length.
    # Using it here previously caused false rejections, since conversation_transcript only
    # ever contains "AI: <question>" lines at the point an individual turn is validated (the
    # real per-turn transcript hadn't been merged into it yet), making every answer look
    # like an empty repeat-the-question.
    text_source = "transcribed_text"
    text = (
        getattr(speaking_input, "transcribed_text", None)
        or (state.get("metadata") or {}).get("transcription_text")
        or ""
    )
    text = text.strip()
    transcript_word_count = word_count(text)

    question_text: Optional[str] = speaking_input.question.question_text if speaking_input.question else None
    asset_context: Optional[str] = None
    question_type: Optional[str] = speaking_input.question.question_type if speaking_input.question else None
    duration_seconds: Optional[int] = speaking_input.question.duration_seconds if speaking_input.question else None
    min_response_seconds: Optional[int] = speaking_input.question.min_response_seconds if speaking_input.question else None
    actual_response_seconds = _coerce_positive_float(metadata.get("actual_response_seconds"))
    expected_min_response_seconds = _coerce_positive_float(min_response_seconds)
    mode: Optional[str] = getattr(speaking_input, "mode", None)
    difficulty_level: Optional[str] = speaking_input.question.difficulty_level if speaking_input.question else None
    off_topic_examples: Optional[str] = (
        speaking_input.question.evaluation_guide.off_topic_examples
        if speaking_input.question and speaking_input.question.evaluation_guide
        else None
    )

    if speaking_input.question and speaking_input.question.asset:
        asset = speaking_input.question.asset
        asset_details = asset.transcript or asset.description or asset.alt_text
        if asset_details:
            asset_context = (
                f"Asset type: {asset.type}\n"
                f"Asset details: {asset_details}\n"
                "(Note: this is objective/factual information about the asset's content, "
                "NOT a model answer or the single correct interpretation. If the question "
                "asks the student to describe feelings, meaning, or their opinion about the "
                "asset, a DIFFERENT interpretation than what's described above is still VALID "
                "as long as it genuinely engages with the asset's actual content and is "
                "reasonably argued -- do not mark it off-topic or incorrect just because it "
                "diverges from this description.)"
            )

    rule_results: List[Any] = []

    # ------------------------------------------------------------------
    # Pure logic rule 1: no_speech (always reject)
    # ------------------------------------------------------------------

    if transcript_word_count == 0:
        rule_results.append(normalize_rule_result({
            "rule_id": "audio.no_speech",
            "category": "audio",
            "severity": "critical",
            "action": "reject_or_zero",
            "blocking": True,
            "message": "No speech detected in audio.",
            "evidence": {
                "word_count": 0,
                "actual_response_seconds": actual_response_seconds,
                "expected_min_response_seconds": expected_min_response_seconds,
            },
            "target_criteria": [],
        }))

    # ------------------------------------------------------------------
    # Pure logic rule 1b: wrong_language deterministic backstop
    # ------------------------------------------------------------------

    code_switching_ratio: Optional[float] = None
    if transcript_word_count > 0:
        code_switching_ratio = compute_code_switching_ratio(text)
        if code_switching_ratio > WRONG_LANGUAGE_RATIO_THRESHOLD:
            rule_results.append(normalize_rule_result({
                "rule_id": "language.wrong_language_full",
                "category": "language",
                "severity": "critical",
                "action": "reject_or_zero",
                "blocking": True,
                "message": (
                    f"{round(code_switching_ratio * 100)}% of words are outside the target "
                    "language (deterministic code-switch ratio, not an LLM judgment)."
                ),
                "evidence": {
                    "word_count": transcript_word_count,
                    "code_switching_ratio": code_switching_ratio,
                    "threshold": WRONG_LANGUAGE_RATIO_THRESHOLD,
                    "source": "deterministic",
                },
                "target_criteria": [],
            }))

    # ------------------------------------------------------------------
    # Pure logic rule 2: too_short (conditional)
    # ------------------------------------------------------------------

    if mode != SpeakingMode.SCRIPTED and transcript_word_count > 0:
        expected_min = get_expected_min_words(
            question_type,
            duration_seconds,
            min_response_seconds=min_response_seconds,
        )
        word_ratio = _round_ratio(float(transcript_word_count), float(expected_min))
        duration_ratio = _round_ratio(actual_response_seconds, expected_min_response_seconds)
        if transcript_word_count < expected_min:
            if STRICT_ZERO_ON_TOO_SHORT:
                rule_results.append(normalize_rule_result({
                    "rule_id": "answer_length.too_short",
                    "category": "length",
                    "severity": "critical",
                    "action": "reject_or_zero",
                    "blocking": True,
                    "message": f"Answer has {transcript_word_count} words, expected at least {expected_min}.",
                    "evidence": {
                        "word_count": transcript_word_count,
                        "expected_min_words": expected_min,
                        "word_ratio": word_ratio,
                        "actual_response_seconds": actual_response_seconds,
                        "expected_min_response_seconds": expected_min_response_seconds,
                        "duration_ratio": duration_ratio,
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
                    "message": f"Answer has {transcript_word_count} words, expected at least {expected_min}. Score will be penalized.",
                    "evidence": {
                        "word_count": transcript_word_count,
                        "expected_min_words": expected_min,
                        "word_ratio": word_ratio,
                        "actual_response_seconds": actual_response_seconds,
                        "expected_min_response_seconds": expected_min_response_seconds,
                        "duration_ratio": duration_ratio,
                    },
                    "target_criteria": ["grammar", "vocabulary", "coherence"],
                }))

        if (
            actual_response_seconds is not None
            and expected_min_response_seconds is not None
            and actual_response_seconds < expected_min_response_seconds * 0.5
        ):
            rule_results.append(normalize_rule_result({
                "rule_id": "answer_duration.too_short",
                "category": "duration",
                "severity": "medium",
                "action": "score_with_penalty",
                "blocking": False,
                "message": (
                    f"Recorded response lasted {round(actual_response_seconds, 1)}s, "
                    f"expected at least {round(expected_min_response_seconds, 1)}s."
                ),
                "evidence": {
                    "word_count": transcript_word_count,
                    "expected_min_words": expected_min,
                    "word_ratio": word_ratio,
                    "actual_response_seconds": actual_response_seconds,
                    "expected_min_response_seconds": expected_min_response_seconds,
                    "duration_ratio": _round_ratio(actual_response_seconds, expected_min_response_seconds),
                },
                "target_criteria": ["fluency", "grammar", "vocabulary", "coherence"],
            }))

    # If pure logic already has a blocking reject, skip LLM call
    has_blocking_reject = any(
        _rule_attr(r, "blocking") and _rule_attr(r, "action") == "reject_or_zero"
        for r in rule_results
    )

    # ------------------------------------------------------------------
    # LLM rules (semantic understanding)
    # ------------------------------------------------------------------

    if not has_blocking_reject and transcript_word_count > 0:
        try:
            llm_response = _call_llm(
                text=text,
                question_text=question_text,
                question_type=question_type,
                asset_context=asset_context,
                mode=mode,
                difficulty_level=difficulty_level,
                word_count=transcript_word_count,
                actual_response_seconds=actual_response_seconds,
                expected_min_response_seconds=expected_min_response_seconds,
                off_topic_examples=off_topic_examples,
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
        transcript_source=text_source,
        transcript_word_count=transcript_word_count,
    )

    logger.info(
        "[eval:validity] done answer_id=%s turn=%s action=%s rule_ids=%s transcript=%r",
        answer_id, turn_order, getattr(validity, "action", None),
        [_rule_attr(r, "rule_id") for r in rule_results], text[:200],
    )

    return {
        **state,
        "validity": validity,
        "status": "processing",
        "error": None,
    }
