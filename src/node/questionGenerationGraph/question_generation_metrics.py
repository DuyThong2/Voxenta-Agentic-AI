import statistics
from dataclasses import asdict
from typing import Any

from node.questionGenerationGraph.constants import (
    EMBEDDING_MODEL,
    HARD_CAP,
    MODEL,
)
from node.questionGenerationGraph.question_generation_graph_helper import (
    TokenCall,
    percentile,
    rank_distribution,
)


def build_summary(pipeline) -> dict[str, Any]:
    calls = [asdict(call) for call in pipeline.all_tokens]
    totals = {
        "input": sum(call.input for call in pipeline.all_tokens),
        "output": sum(call.output for call in pipeline.all_tokens),
        "reasoning": sum(call.reasoning for call in pipeline.all_tokens),
        "cached_input": sum(
            call.cached_input for call in pipeline.all_tokens
        ),
    }
    role_totals: dict[str, dict[str, int]] = {}
    for call in pipeline.all_tokens:
        aggregate = role_totals.setdefault(
            call.role,
            {
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cached_input": 0,
            },
        )
        aggregate["input"] += call.input
        aggregate["output"] += call.output
        aggregate["reasoning"] += call.reasoning
        aggregate["cached_input"] += call.cached_input
    return {
        "hard_cap": HARD_CAP,
        "accepted": pipeline.accepted_count,
        "candidate_total": pipeline.candidate_total,
        "success_ratio_p": _ratio(
            pipeline.accepted_count,
            pipeline.candidate_total,
        ),
        "evaluator_rejection_ratio": _ratio(
            pipeline.evaluator_rejected,
            pipeline.candidate_total,
        ),
        "average_editor_rounds": (
            statistics.mean(pipeline.editor_rounds)
            if pipeline.editor_rounds
            else 0.0
        ),
        "evaluator_comparison": {
            "total": pipeline.comparison_total,
            "different": pipeline.comparison_different,
            "difference_ratio": _ratio(
                pipeline.comparison_different,
                pipeline.comparison_total,
            ),
        },
        "cosine": {
            "count": len(pipeline.cosines),
            "p95": percentile(pipeline.cosines, 0.95),
            "max": max(pipeline.cosines, default=0.0),
        },
        "tokens": {
            "totals": totals,
            "by_role": role_totals,
            "calls": calls,
        },
        "difficulty_rank_distribution": rank_distribution(
            pipeline.accepted_questions
        ),
        "persisted": pipeline.persist,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": MODEL,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
