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
{SAFETY_CONSTRAINTS}
Do not assign a total quality score. Return concrete violation codes.

TOPIC
{topic[0]}

CANDIDATES TO EVALUATE
{json.dumps([item.model_dump() for item in candidates], ensure_ascii=False)}"""
