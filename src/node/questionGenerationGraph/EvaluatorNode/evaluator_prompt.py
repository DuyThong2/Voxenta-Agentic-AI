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
    band_ladder: str | None = None,
) -> str:
    """Prompt cham ung vien.

    `band_ladder` do Java gui xuong (dung tu framework_result_bands cua truong). Rong thi lui
    ve hang so BAND_LADDER -- hang so do viet cung BAC_1..BAC_6 kieu VSTEP nen chi dung lam
    fallback cho pipeline nghien cuu/goi tay, khong dung cho duong that.
    """
    ladder = band_ladder or BAND_LADDER
    return f"""{ladder}

EVALUATION RULES
Return one verdict per candidate. accepted=true only when violations is empty.
Check topic fit, target construct, target sub-attribute, neutral access, internal consistency,
all six evaluation-guide fields, and whether difficulty_features imply rank near {target_rank}.
The target_sub_attribute taxonomy is closed. PRONUNCIATION and FLUENCY require null;
null is correct for those two constructs and must never be reported as missing. GRAMMAR allows
tense_control, complex_clause_control. VOCABULARY allows no sub-attribute. COHERENCE allows
weak_progression, limited_support.

TENSE ANCHOR - judge the question, not the label
Each candidate declares a `target_tense`. Reject it (violation code TENSE_NOT_FORCED) unless
the wording makes that tense the only natural way to answer. The test is concrete: imagine a
learner answering in a different tense - if that answer would still be correct English and on
topic, the anchor failed.
- "Tell me about your school's history" declaring PAST: REJECT. "My school is old and has a
  big garden" answers it in the present and is perfectly correct.
- "What did your class do for the school festival last year?" declaring PAST: accept.
- A CONDITIONAL candidate must describe something unreal or not yet true, not merely a
  future plan.
Also check `suggested_ideas`: they are shown to the learner as a starting point, so ideas
written in a different tense pull against the anchor the question set.
{SAFETY_CONSTRAINTS}
Do not assign a total quality score. Return concrete violation codes.

TOPIC
{topic[0]}

CANDIDATES TO EVALUATE
{json.dumps([item.model_dump() for item in candidates], ensure_ascii=False)}"""
