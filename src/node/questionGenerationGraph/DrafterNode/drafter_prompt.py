import json

from node.questionGenerationGraph.constants import (
    BAND_LADDER,
    DRAFTER_CANDIDATES,
    SAFETY_CONSTRAINTS,
    build_band_ladder,
)
from schemas.question_generation import (
    PRACTICE_TYPE_SECONDS,
    feature_profiles_for_raw,
    raw_for_rank,
)


def _difficulty_block(target_rank: int, band_count: int) -> str:
    """Nói độ khó bằng BỐN CẦN GẠT, không bằng một con số.

    Trước đây chỗ này chỉ có đúng một dòng: "Target cognitive rank: approximately {n}".
    Con số đó vô nghĩa với người soạn câu -- nó không biết bậc 4 khác bậc 3 ở chỗ nào, và
    `difficulty_rank` thì lại được TÍNH từ bốn đặc trưng Robinson mà chính nó tự khai. Tức
    là xin một con số, nhận về một con số do bên kia tự chấm: vòng tròn, không có neo nào.

    Bậc 6 chỉ có 2/60 tổ hợp đặc trưng, bậc 1 cũng vậy. Không nói ra thì soạn tự do trúng
    khoảng 3% -- đó là lý do kho không bao giờ tự có câu ở hai đầu thang.

    Mỗi ứng viên nhận một tổ hợp KHÁC nhau trong cùng bậc (khi bậc đó có nhiều hơn một).
    Đây mới là "rải độ khó" đúng nghĩa: không rải sang bậc khác -- học sinh đã chọn bậc rồi
    -- mà rải các cách viết khác nhau cho cùng một bậc.
    """
    raw = raw_for_rank(target_rank, band_count)
    profiles = feature_profiles_for_raw(raw)
    lines = []
    for index in range(DRAFTER_CANDIDATES):
        profile = profiles[index % len(profiles)]
        elements = "at least 4" if profile.num_elements >= 4 else "2 or 3"
        lines.append(
            f"- Candidate {index + 1}: "
            f"here_and_now={str(profile.here_and_now).lower()}, "
            f"num_elements={elements}, "
            f"reasoning_type={profile.reasoning_type}, "
            f"abstractness={profile.abstractness}"
        )
    repeated = len(profiles) < DRAFTER_CANDIDATES
    return f"""DIFFICULTY - set these four dials, do not guess a number
The band above says what a learner at this level can do. These four dials say what makes THIS
question that hard. They are the exact inputs the difficulty score is computed from, so a
question that matches its dials lands on the intended band by construction.

{chr(10).join(lines)}

What each dial means when you write:
- here_and_now=true: about what the student can see or is living right now.
  false: another time, another place, or someone else's situation.
- num_elements: how many distinct things a complete answer must cover.
- reasoning_type: description (just tell) < comparison / causal (weigh, explain why)
  < intentional / hypothetical (motives, or what-if that has not happened).
- abstractness: concrete_personal (own life, tangible) / mixed / abstract (ideas, concepts).

Report difficulty_features exactly as assigned. If a dial genuinely cannot fit the topic
without an unnatural question, write the natural question and report what you actually
did: a mismatch is caught downstream, a forced question is not.
{"Fewer distinct dial settings exist at this band than candidates, so some repeat; vary the wording and angle instead." if repeated else ""}"""


def build_drafter_prompt(
    topic: tuple[str, str, str],
    criterion: tuple[str, str | None],
    target_rank: int,
    band_ladder: list | None = None,
    band_count: int = 6,
) -> str:
    # Dai thoi luong lay THANG tu PRACTICE_TYPE_SECONDS -- cung bang ma validator dung de
    # kep. Viet tay lai o day la mo duong cho prompt va bo kep noi hai dieu khac nhau.
    ranges_block = "\n".join(
        f"  * {name}: min {mn[0]}-{mn[1]}s, max {mx[0]}-{mx[1]}s"
        for name, (mn, mx) in PRACTICE_TYPE_SECONDS.items()
    )
    ladder = build_band_ladder(band_ladder, band_count) if band_ladder else BAND_LADDER
    return f"""Generate exactly {DRAFTER_CANDIDATES} different English speaking questions.
Topic: {topic[0]}
Target construct: {criterion[0]}
Target sub-attribute: {json.dumps(criterion[1])}

{ladder}

Target band: {target_rank}.

{_difficulty_block(target_rank, band_count)}

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

Pick the type that carries the dials above. A `description` + `here_and_now` + `2 or 3`
question is a SHORT_ANSWER or a concrete DESCRIPTION; `hypothetical` or `intentional` with
`abstract` needs OPINION or a developed LONG_ANSWER, because the learner has to sustain an
argument. Never ask a beginner to defend a position about something outside their own
experience.

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

TARGET SUB-ATTRIBUTE - write the question so the answer HAS to exercise it
The taxonomy is closed. Return exactly the value shown above; never invent another.
- PRONUNCIATION, FLUENCY, VOCABULARY: null. Nothing finer can be targeted by wording.
- GRAMMAR / `tense_control`: pin the question to a definite time frame, so a correct
  answer cannot be given in a single default tense. Name the moment ("last summer",
  "by the time you finish school"), or ask the student to contrast two moments.
- GRAMMAR / `complex_clause_control`: ask for cause, condition or consequence, so the
  answer needs subordinate clauses ("because...", "if... then...", "even though...").
  A question answerable as a flat list does not target this.
- COHERENCE / `weak_progression`: ask for something that unfolds in order - steps, a
  sequence of events, how something changed over time. The answer must go somewhere,
  not just enumerate.
- COHERENCE / `limited_support`: ask the student to take a position AND back it with
  concrete reasons or examples. A bare preference question does not target this.

Say in `scoring_hints` what a good answer must do for this sub-attribute - later stages
read that field, and it is the only place they learn what this question was aiming at."""
