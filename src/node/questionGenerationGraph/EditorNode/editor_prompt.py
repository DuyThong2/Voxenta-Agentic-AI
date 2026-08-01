import json

from node.questionGenerationGraph.constants import SAFETY_CONSTRAINTS
from schemas.question_generation import (
    CandidateVerdict,
    PracticeQuestionCandidate,
)


def build_editor_prompt(
    candidate: PracticeQuestionCandidate,
    verdict: CandidateVerdict,
    topic: tuple[str, str, str],
    target_rank: int,
    round_number: int,
) -> str:
    return f"""Edit this candidate to remove every listed violation.
Keep candidate_id unchanged. Do not return difficulty_rank or followup_questions.
Topic: {topic[0]}. Target rank: {target_rank}. Editor round: {round_number}.
Violations: {json.dumps(verdict.violations, ensure_ascii=False)}
Candidate: {candidate.model_dump_json()}
PRONUNCIATION and FLUENCY require target_sub_attribute=null. Preserve that null value.
{SAFETY_CONSTRAINTS}"""
