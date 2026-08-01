from node.questionGenerationGraph.EvaluatorNode.evaluator_prompt import (
    build_evaluator_prompt,
)
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
    verdict_signature,
)
from schemas.question_generation import (
    EvaluationBatch,
    PracticeQuestionCandidate,
)


def evaluate(
    runtime: QuestionGenerationRuntime,
    candidates: list[PracticeQuestionCandidate],
    topic: tuple[str, str, str],
    target_rank: int,
    *,
    mode: str,
    tokens: list,
) -> tuple[EvaluationBatch, dict]:
    return runtime.parsed_call(
        role="evaluator",
        mode=mode,
        effort="high",
        system=(
            "You are a strict independent evaluator. "
            "The six-band ladder at the beginning of the user message is authoritative."
        ),
        prompt=build_evaluator_prompt(candidates, topic, target_rank),
        schema=EvaluationBatch,
        tokens=tokens,
    )


def evaluator_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    separate = {}
    separate_raw = []
    for candidate in state["survivors"]:
        result, raw = evaluate(
            runtime,
            [candidate],
            state["topic"],
            state["target_rank"],
            mode="separate",
            tokens=state["token_calls"],
        )
        verdict = result.verdicts[0]
        separate[verdict.candidate_id] = verdict
        separate_raw.append(raw)

    grouped_raw = None
    comparison_total = 0
    comparison_different = 0
    if state["survivors"]:
        grouped, grouped_raw = evaluate(
            runtime,
            state["survivors"],
            state["topic"],
            state["target_rank"],
            mode="grouped",
            tokens=state["token_calls"],
        )
        grouped_by_id = {
            verdict.candidate_id: verdict for verdict in grouped.verdicts
        }
        for candidate in state["survivors"]:
            grouped_item = grouped_by_id.get(candidate.candidate_id)
            comparison_total += 1
            if (
                grouped_item is None
                or verdict_signature(separate[candidate.candidate_id])
                != verdict_signature(grouped_item)
            ):
                comparison_different += 1

    return {
        "separate_verdicts": separate,
        "comparison_total": comparison_total,
        "comparison_different": comparison_different,
        "evaluator_raw": {
            "separate": separate_raw,
            "grouped": grouped_raw,
            "rejected_before_evaluator": state["rejected"],
            "vector_similarity": {
                candidate_id: runtime.max_similarity(embedding)
                for candidate_id, embedding
                in state["survivor_embeddings"].items()
            },
        },
    }
