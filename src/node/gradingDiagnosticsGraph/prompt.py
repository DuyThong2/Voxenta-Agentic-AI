from utils.criterion_diagnostics import ALLOWED_WEAKNESS_LABELS


def build_grading_diagnostics_prompt(feedback_summary: str, applicable_criteria: list[str]) -> str:
    criteria_block = "\n".join(
        f'- "{criterion}": {sorted(ALLOWED_WEAKNESS_LABELS[criterion])}'
        for criterion in applicable_criteria
    )

    return f"""A teacher just hand-graded one speaking-exam answer and wrote this free-text note
about it (in Vietnamese):

\"\"\"{feedback_summary}\"\"\"

For each criterion below, decide which of its FIXED weakness labels (if any) this note supports.
Only pick a label if the note gives clear evidence for it -- when in doubt, pick nothing for that
criterion. Do not invent labels outside the given lists. For each label you pick, quote the short
part of the note (in Vietnamese, verbatim) that supports it as evidence_span.

Criteria and their allowed labels:
{criteria_block}

Return one entry per criterion listed above (with an empty labels list if nothing applies)."""
