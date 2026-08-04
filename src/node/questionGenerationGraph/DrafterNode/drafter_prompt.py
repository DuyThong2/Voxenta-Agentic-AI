import json

from node.questionGenerationGraph.constants import (
    DRAFTER_CANDIDATES,
    SAFETY_CONSTRAINTS,
)
from schemas.question_generation import PRACTICE_TYPE_SECONDS


def build_drafter_prompt(
    topic: tuple[str, str, str],
    criterion: tuple[str, str | None],
    target_rank: int,
) -> str:
    # Dai thoi luong lay THANG tu PRACTICE_TYPE_SECONDS -- cung bang ma validator dung de
    # kep. Viet tay lai o day la mo duong cho prompt va bo kep noi hai dieu khac nhau.
    ranges_block = "\n".join(
        f"  * {name}: min {mn[0]}-{mn[1]}s, max {mx[0]}-{mx[1]}s"
        for name, (mn, mx) in PRACTICE_TYPE_SECONDS.items()
    )
    return f"""Generate exactly {DRAFTER_CANDIDATES} different English speaking questions.
Topic: {topic[0]}
Target construct: {criterion[0]}
Target sub-attribute: {json.dumps(criterion[1])}
Target cognitive rank: approximately {target_rank}; fill difficulty_features honestly.

{SAFETY_CONSTRAINTS}

Use verbalized sampling internally: consider varied approaches, then return three candidates.
Do not return difficulty_rank. Do not return followup_questions.

Question type — pick the one that fits what you are actually asking for:
- `SHORT_ANSWER`: one fact, preference, or short reason. A couple of sentences is a
  complete answer.
- `LONG_ANSWER`: needs several connected sentences — steps, reasons, or a small story.
- `DESCRIPTION`: describe a place, person, object, routine, or picture-like scene in detail.
- `OPINION`: take a position and defend it, or weigh two sides.

How many things you may ask for depends on the type, and this is a hard rule:
- `SHORT_ANSWER`: exactly ONE ask. One question mark, no trailing "Say also...", "and
  explain why", or "then tell me...". If you find yourself adding a second clause so the
  answer will be long enough, you picked the wrong type -- switch to LONG_ANSWER instead
  of stretching a short question.
- `LONG_ANSWER`, `DESCRIPTION`, `OPINION`: two or three sub-prompts are expected and
  welcome (real speaking exams do exactly this), but they must be facets of ONE subject,
  never two unrelated questions bolted together.

Match the type to the target cognitive rank. A rank-1 learner should get SHORT_ANSWER or a
concrete DESCRIPTION about their own life; OPINION and abstract DESCRIPTION belong at higher
ranks, where the learner can sustain argument. Never ask a beginner to defend a position about
something outside their own experience.

Response-time window, in seconds of actual speech:
- min_response_seconds is the FLOOR: below this the answer cannot have covered the
  question, so the tutor keeps asking follow-ups. Set it to the shortest answer you
  would still accept as complete.
- max_response_seconds is the CEILING for the whole exchange, follow-up turns included.
- Both count only the student speaking; thinking time is not counted, and there is no
  preparation countdown -- the student starts when they are ready.
- Stay inside the window allowed for the type you chose:
{ranges_block}
  These are hard limits; a value outside them is clamped and your intent is lost.
- Within the allowed window, scale to how much the question actually asks for.
Each evaluation guide must have all six non-empty fields.
The target_sub_attribute taxonomy is closed:
- PRONUNCIATION and FLUENCY: null only.
- GRAMMAR: sv_agreement, tense_control, complex_clause_control,
  third_person_s_omission, article_use, word_form.
- VOCABULARY: limited_range, repetition, weak_collocation.
- COHERENCE: weak_progression, limited_support, connector_overuse, topic_drift.
Return exactly the target sub-attribute shown above; never invent another value."""
