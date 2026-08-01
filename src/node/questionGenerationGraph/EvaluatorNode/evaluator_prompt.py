import json

from node.questionGenerationGraph.constants import (
    BAND_LADDER,
    SAFETY_CONSTRAINTS,
)
from schemas.question_generation import PracticeQuestionCandidate


def build_evaluator_prompt(
    candidates: list[PracticeQuestionCandidate],
    topic: tuple[str, str, str],
    target_rank: int,
) -> str:
    return f"""{BAND_LADDER}

EVALUATION RULES
Return one verdict per candidate. accepted=true only when violations is empty.
Check topic fit, target construct, target sub-attribute, neutral access, internal consistency,
all six evaluation-guide fields, and whether difficulty_features imply rank near {target_rank}.
The target_sub_attribute taxonomy is closed. PRONUNCIATION and FLUENCY require null;
null is correct for those two constructs and must never be reported as missing. GRAMMAR allows
sv_agreement, tense_control, complex_clause_control, third_person_s_omission, article_use,
word_form. VOCABULARY allows limited_range, repetition, weak_collocation. COHERENCE allows
weak_progression, limited_support, connector_overuse, topic_drift.
{SAFETY_CONSTRAINTS}
Do not assign a total quality score. Return concrete violation codes.

TOPIC
{topic[0]}

CANDIDATES TO EVALUATE
{json.dumps([item.model_dump() for item in candidates], ensure_ascii=False)}"""
