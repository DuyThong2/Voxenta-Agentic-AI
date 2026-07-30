import json

from node.questionGenerationGraph.constants import (
    DRAFTER_CANDIDATES,
    SAFETY_CONSTRAINTS,
)


def build_drafter_prompt(
    topic: tuple[str, str, str],
    criterion: tuple[str, str | None],
    target_rank: int,
) -> str:
    return f"""Generate exactly {DRAFTER_CANDIDATES} different English speaking questions.
Topic: {topic[0]}
Target construct: {criterion[0]}
Target sub-attribute: {json.dumps(criterion[1])}
Target cognitive rank: approximately {target_rank}; fill difficulty_features honestly.

{SAFETY_CONSTRAINTS}

Use verbalized sampling internally: consider varied approaches, then return three candidates.
Do not return difficulty_rank. Do not return followup_questions.
Each evaluation guide must have all six non-empty fields.
The target_sub_attribute taxonomy is closed:
- PRONUNCIATION and FLUENCY: null only.
- GRAMMAR: sv_agreement, tense_control, complex_clause_control,
  third_person_s_omission, article_use, word_form.
- VOCABULARY: limited_range, repetition, weak_collocation.
- COHERENCE: weak_progression, limited_support, connector_overuse, topic_drift.
Return exactly the target sub-attribute shown above; never invent another value."""
