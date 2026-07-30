import json

from schemas.question_generation import PracticeQuestionCandidate


def build_refiner_prompt(
    candidates: list[PracticeQuestionCandidate],
    topic: tuple[str, str, str],
    target_rank: int,
) -> str:
    return f"""Polish these {len(candidates)} independent accepted questions in one pass.
Preserve candidate IDs, constructs, sub-attributes, difficulty features, time budgets, and
evaluation-guide meaning. Improve only clarity and natural English. Do not add
followup_questions or difficulty_rank. Topic: {topic[0]}. Target rank: {target_rank}.
Candidates: {json.dumps([item.model_dump() for item in candidates], ensure_ascii=False)}"""
