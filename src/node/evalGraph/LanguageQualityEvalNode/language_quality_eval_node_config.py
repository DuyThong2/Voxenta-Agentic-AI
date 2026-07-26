"""Combined LLM evaluation for grammar, vocabulary, and coherence."""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from node.evalGraph.LanguageQualityEvalNode.language_quality_eval_prompt import (
    SYSTEM_PROMPT,
)
from node.state_models import SpeakingInput
from schemas.enums import SpeakingMode
from schemas.scoring import CriterionScore
from utils.confidence_utils import run_multi_criterion_consensus_judgment

logger = logging.getLogger(__name__)


def _format_asset_context(speaking_input: SpeakingInput) -> list[str]:
    question = speaking_input.question
    if not question or not question.asset:
        return []

    asset = question.asset
    parts = ["\n## Question Asset"]
    if asset.type:
        parts.append(f"Asset type: {asset.type}")
    if asset.transcript:
        parts.append(f"Asset transcript: {asset.transcript}")
    elif asset.description:
        parts.append(f"Asset description: {asset.description}")
    elif asset.alt_text:
        parts.append(f"Asset alt text: {asset.alt_text}")
    if asset.transcript or asset.description or asset.alt_text:
        parts.append(
            "This is factual context about the asset, not a model answer. A different "
            "interpretation remains valid when it engages with the asset and is reasonably argued."
        )
    return parts


def _select_transcript(speaking_input: SpeakingInput) -> Tuple[Optional[str], str]:
    if speaking_input.transcribed_text:
        return speaking_input.transcribed_text, "transcribed_text"
    if speaking_input.corrected_transcript:
        return speaking_input.corrected_transcript, "corrected_transcript"
    if speaking_input.reference_text:
        return speaking_input.reference_text, "reference_text_fallback"
    return None, "missing"


def _build_scoring_metadata(source: str, mode: Optional[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"language_scoring_text_source": source}
    if source == "reference_text_fallback":
        metadata.update(
            {
                "language_scoring_note": (
                    "No transcribed_text or corrected_transcript available. Fell back to "
                    "reference_text; language scores are diagnostic only."
                ),
                "language_scoring_status": "diagnostic_only",
            }
        )
    if mode == SpeakingMode.SCRIPTED and source != "transcribed_text":
        metadata.update(
            {
                "language_scoring_note": (
                    "Scripted mode: LLM language scores are diagnostic only. Official scoring "
                    "uses Azure pronunciation assessment."
                ),
                "language_scoring_status": "diagnostic_only",
            }
        )
    return metadata


def _build_question_context(speaking_input: SpeakingInput) -> str:
    parts: list[str] = []
    question = speaking_input.question
    topic = speaking_input.topic

    if question and question.question_text:
        parts.append(f'Question: "{question.question_text}"')
    if question and question.question_type:
        parts.append(f"Question type: {question.question_type}")
    if question and question.difficulty_level:
        parts.append(f"Difficulty: {question.difficulty_level}")
    if question and question.duration_seconds is not None:
        parts.append(f"Expected duration: {question.duration_seconds}s")
    if (
        question
        and question.min_response_seconds is not None
        and question.max_response_seconds is not None
    ):
        parts.append(
            "Expected response length: "
            f"{question.min_response_seconds}-{question.max_response_seconds}s"
        )
    elif question and question.min_response_seconds is not None:
        parts.append(
            f"Expected response length: at least {question.min_response_seconds}s"
        )
    elif question and question.max_response_seconds is not None:
        parts.append(
            f"Expected response length: up to {question.max_response_seconds}s"
        )

    if topic and topic.topic_name:
        parts.append(f"Topic: {topic.topic_name}")
    if topic and topic.topic_description:
        parts.append(f"Topic description: {topic.topic_description}")

    if question and question.evaluation_guide:
        guide = question.evaluation_guide
        parts.append("\n## Evaluation Guide")
        if guide.expected_content:
            parts.append(f"Expected content: {guide.expected_content}")
        if guide.key_points:
            parts.append(f"Key points: {guide.key_points}")
        if guide.acceptable_responses:
            parts.append(f"Acceptable responses: {guide.acceptable_responses}")
        if guide.off_topic_examples:
            parts.append(f"Off-topic examples: {guide.off_topic_examples}")
        if guide.scoring_hints:
            parts.append(f"Scoring hints: {guide.scoring_hints}")
        if guide.common_mistakes:
            parts.append(
                "Common mistakes (supporting context, not exhaustive): "
                f"{guide.common_mistakes}"
            )

    parts.extend(_format_asset_context(speaking_input))
    return "\n".join(parts) if parts else "No question context provided."


def _find_framework(speaking_input: SpeakingInput, criterion_key: str):
    return next(
        (
            framework
            for framework in (speaking_input.criteria_frameworks or [])
            if framework.criterion_key == criterion_key
        ),
        None,
    )


def _build_framework_context(
    speaking_input: SpeakingInput,
    criterion_key: str,
    title: str,
) -> str:
    framework = _find_framework(speaking_input, criterion_key)
    if framework is None:
        return (
            f"## {title} Scoring Framework\n"
            "Score range for this criterion: 0-100"
        )

    score_range = (
        f"{framework.rubric_min_score}-{framework.rubric_max_score}"
    )
    lines = [f"## {title} Scoring Framework"]
    if framework.framework_criterion_name:
        lines.append(
            f"Framework criterion: {framework.framework_criterion_name}"
        )
    if framework.framework_criterion_description:
        lines.append(f"Definition: {framework.framework_criterion_description}")
    if framework.target_band_only:
        target_label = framework.target_band_label or framework.target_band_code
        lines.extend(
            [
                f"Target proficiency level: {target_label}",
                (
                    f"Scoring scope: assign a score only within {score_range} "
                    "for how fully this answer "
                    "satisfies the target-level descriptor below. This is NOT a "
                    "placement score across all six framework levels: the minimum "
                    "does not mean Bậc 1 and the maximum does not mean Bậc 6. Do "
                    "not evaluate or assign any other proficiency level."
                ),
            ]
        )
    lines.append(f"Score range for this criterion: {score_range}")
    for band in sorted(framework.bands, key=lambda item: item.score_min):
        line = f"- {band.code} ({band.score_min}-{band.score_max})"
        if band.label:
            line += f" [{band.label}]"
        if band.descriptor:
            line += f": {band.descriptor}"
        lines.append(line)
        if band.positive_signals:
            lines.append(
                f"  Positive signals: {', '.join(band.positive_signals)}"
            )
        if band.negative_signals:
            lines.append(
                f"  Negative signals: {', '.join(band.negative_signals)}"
            )
    return "\n".join(lines)


def _score_range(
    speaking_input: SpeakingInput,
    criterion_key: str,
) -> Tuple[float, float]:
    framework = _find_framework(speaking_input, criterion_key)
    if framework is None:
        return 0.0, 100.0
    return framework.rubric_min_score, framework.rubric_max_score


def _scale_percentage_cap(
    speaking_input: SpeakingInput,
    criterion_key: str,
    raw_cap: Any,
) -> float:
    """Map a 0-100 evidence cap to the criterion's authored score range."""
    try:
        percentage = min(100.0, max(0.0, float(raw_cap)))
    except (TypeError, ValueError):
        percentage = 100.0
    score_min, score_max = _score_range(speaking_input, criterion_key)
    return round(score_min + (percentage / 100.0) * (score_max - score_min), 2)


def build_user_prompt(
    speaking_input: SpeakingInput,
    transcript: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    mode = speaking_input.mode or SpeakingMode.UNSCRIPTED
    parts = [
        "## Question Context",
        _build_question_context(speaking_input),
        "",
        _build_framework_context(speaking_input, "grammar", "Grammar"),
        "",
        _build_framework_context(speaking_input, "vocabulary", "Vocabulary"),
        "",
        _build_framework_context(speaking_input, "coherence", "Coherence"),
        "",
        "## Student Answer",
        f"Mode: {mode}",
        f'Transcript: "{transcript}"',
    ]

    if speaking_input.conversation_transcript:
        parts.extend(
            [
                "",
                "## Dialogue Context (for coherence context only)",
                (
                    "Use this only to understand what was asked. Never grade or quote AI lines "
                    "as student speech."
                ),
                speaking_input.conversation_transcript,
            ]
        )

    if metrics:
        grammar_cap = _scale_percentage_cap(
            speaking_input,
            "grammar",
            metrics.get("grammar_range_cap"),
        )
        vocabulary_cap = _scale_percentage_cap(
            speaking_input,
            "vocabulary",
            metrics.get("lexical_range_cap"),
        )
        coherence_cap = _scale_percentage_cap(
            speaking_input,
            "coherence",
            metrics.get("coherence_cap"),
        )
        parts.extend(
            [
                "",
                "## Answer Length Metrics",
                f"Word count: {metrics.get('word_count')}",
                f"Sentence count: {metrics.get('sentence_count')}",
                f"Length category: {metrics.get('length_category')}",
                f"Expected min words: {metrics.get('expected_min_words')}",
                f"Grammar range cap (criterion scale): {grammar_cap}",
                f"Lexical range cap (criterion scale): {vocabulary_cap}",
                f"Coherence cap (criterion scale): {coherence_cap}",
                (
                    "Code-switching ratio (non-English words / total words): "
                    f"{metrics.get('code_switching_ratio')}"
                ),
            ]
        )

    if mode == SpeakingMode.SCRIPTED:
        parts.append(
            "\nThis is scripted/read-aloud. All three language-quality judgments are "
            "diagnostic only."
        )
    return "\n".join(parts)


def _to_criterion(response: Dict[str, Any]) -> CriterionScore:
    return CriterionScore(
        score=response["score"],
        subscores=response.get("subscores", {}),
        note=response.get(
            "note",
            "Evaluated by LLM from the student's transcript.",
        ),
    )


def language_quality_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    speaking_input = state.get("speaking_input")
    if speaking_input is None:
        return {
            "metadata": {
                "language_quality_error": (
                    "speaking_input is required for language_quality_eval_node"
                )
            }
        }

    transcript, source = _select_transcript(speaking_input)
    if transcript is None:
        return {
            "metadata": {
                "language_quality_error": (
                    "No transcript available for language quality evaluation"
                )
            }
        }

    answer_id = getattr(speaking_input, "answer_id", None)
    turn_order = (state.get("metadata") or {}).get("turn_order")
    logger.info(
        "[eval:language_quality] calling 3 parallel O-C-O passes answer_id=%s turn=%s",
        answer_id,
        turn_order,
    )

    try:
        judgments = run_multi_criterion_consensus_judgment(
            SYSTEM_PROMPT,
            build_user_prompt(
                speaking_input,
                transcript,
                state.get("answer_length_metrics"),
            ),
            transcript,
            {
                "grammar": _score_range(speaking_input, "grammar"),
                "vocabulary": _score_range(speaking_input, "vocabulary"),
                "coherence": _score_range(speaking_input, "coherence"),
            },
        )
        grammar = judgments["grammar"]
        vocabulary = judgments["vocabulary"]
        coherence = judgments["coherence"]
        metadata = {
            **_build_scoring_metadata(source, speaking_input.mode),
            "grammar_confidence": grammar.confidence,
            "vocabulary_confidence": vocabulary.confidence,
            "coherence_confidence": coherence.confidence,
            "grammar_score_delta": grammar.score_delta_on_ten_point_scale,
            "lexical_score_delta": vocabulary.score_delta_on_ten_point_scale,
            "coherence_score_delta": coherence.score_delta_on_ten_point_scale,
            "language_quality_diagnostics": {
                "grammar": {
                    "evidence_spans": grammar.response.get("evidence_spans", []),
                    "weakness_labels": grammar.response.get("weakness_labels", []),
                    "recommendation_tag": grammar.response.get(
                        "recommendation_tag",
                        "",
                    ),
                },
                "vocabulary": {
                    "evidence_spans": vocabulary.response.get(
                        "evidence_spans",
                        [],
                    ),
                    "weakness_labels": vocabulary.response.get(
                        "weakness_labels",
                        [],
                    ),
                    "recommendation_tag": vocabulary.response.get(
                        "recommendation_tag",
                        "",
                    ),
                },
                "coherence": {
                    "evidence_spans": coherence.response.get(
                        "evidence_spans",
                        [],
                    ),
                    "weakness_labels": coherence.response.get(
                        "weakness_labels",
                        [],
                    ),
                    "recommendation_tag": coherence.response.get(
                        "recommendation_tag",
                        "",
                    ),
                },
            },
        }
        logger.info(
            (
                "[eval:language_quality] done answer_id=%s turn=%s "
                "grammar=%s vocabulary=%s coherence=%s"
            ),
            answer_id,
            turn_order,
            grammar.response.get("score"),
            vocabulary.response.get("score"),
            coherence.response.get("score"),
        )
        return {
            "grammar_criterion": _to_criterion(grammar.response),
            "lexical_criterion": _to_criterion(vocabulary.response),
            "coherence_criterion": _to_criterion(coherence.response),
            "metadata": metadata,
        }
    except json.JSONDecodeError as exc:
        logger.exception(
            "[eval:language_quality] LLM returned invalid JSON answer_id=%s turn=%s",
            answer_id,
            turn_order,
        )
        return {
            "metadata": {
                "language_quality_error": f"LLM returned invalid JSON: {exc}"
            }
        }
    except Exception as exc:
        logger.exception(
            "[eval:language_quality] failed answer_id=%s turn=%s",
            answer_id,
            turn_order,
        )
        return {
            "metadata": {
                "language_quality_error": (
                    f"language quality evaluation failed: {exc}"
                )
            }
        }
