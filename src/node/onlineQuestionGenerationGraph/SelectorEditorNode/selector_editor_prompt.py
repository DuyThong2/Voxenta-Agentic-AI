from schemas.question_generation import (
    PRACTICE_TYPE_SECONDS,
    PracticeQuestionCandidate,
)


def build_selector_editor_prompt(
    candidates: list[PracticeQuestionCandidate],
    topic: tuple[str, str, str],
    criterion: tuple[str, str | None],
    target_rank: int,
    band_ladder: str,
) -> str:
    """Prompt cho MOT luot: doc ca N ung vien, chon 1, chinh no, chot loai + thoi luong.

    Gop ba viec vao mot luot chu khong tach ra: hoc sinh dang ngoi cho, moi luot goi them la
    them vai giay. Ba viec nay cung can doc ca ba ung vien nen tach ra la doc lai ba lan.
    """
    listing = "\n".join(
        f"""
[{index}] id={candidate.candidate_id}
    prompt_text: {candidate.prompt_text}
    question_type (de xuat): {candidate.question_type}
    thoi luong (de xuat): {candidate.min_response_seconds}-{candidate.max_response_seconds}s
    difficulty_features: {candidate.difficulty_features.model_dump()}
    suggested_ideas: {candidate.suggested_ideas}
"""
        for index, candidate in enumerate(candidates, 1)
    )
    ranges = "\n".join(
        f"- {name}: min {mn[0]}-{mn[1]}s, max {mx[0]}-{mx[1]}s"
        for name, (mn, mx) in PRACTICE_TYPE_SECONDS.items()
    )

    return f"""You are choosing and finishing ONE English speaking-practice question for a
Vietnamese high-school student, from the drafts below.

Topic: {topic[0]}
Target construct: {criterion[0]}
Target sub-attribute: {criterion[1]}
Target band: {target_rank}

Band ladder for this school's framework:
{band_ladder}

Drafts:
{listing}

Do THREE things, in this order.

1. CHOOSE the single draft that best fits band {target_rank} and the target construct. Judge
   fit to the learner's level first, then how naturally a student could answer it. Do not
   default to the first draft.

2. REPAIR the chosen one. Keep its intent; fix awkward phrasing, ambiguity, and anything a
   student at this band would not understand. Keep suggested_ideas useful and concrete.

3. SET question_type and the response window from what the question ACTUALLY asks for, not
   from what the draft guessed:
   - SHORT_ANSWER: one fact, preference, or short reason.
   - LONG_ANSWER: several connected sentences -- steps, reasons, a small story.
   - DESCRIPTION: describe a place, person, object, routine, or scene in detail.
   - OPINION: take a position and defend it, or weigh two sides.
   Allowed windows per type (seconds of actual speech):
{ranges}
   min_response_seconds is the FLOOR: below it the answer cannot have covered the question.
   max_response_seconds is the CEILING for the whole exchange, follow-up turns included.
   A low-band learner should not be given OPINION or an abstract DESCRIPTION.

Return the finished question as a single candidate. Keep the candidate_id of the draft you
chose, so the choice can be traced. Return structured data only."""
