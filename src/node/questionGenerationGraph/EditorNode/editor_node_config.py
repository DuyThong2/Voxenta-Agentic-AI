from node.questionGenerationGraph.constants import MAX_EDITOR_ROUNDS
from node.questionGenerationGraph.EditorNode.editor_prompt import (
    build_editor_prompt,
)
from node.questionGenerationGraph.EvaluatorNode.evaluator_node_config import (
    evaluate,
)
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
)
from schemas.question_generation import PracticeQuestionCandidate


def editor_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    live = []
    rejected = list(state["rejected"])
    raw_calls = []
    rounds_used = []
    evaluator_rejected = 0
    for candidate in state["survivors"]:
        verdict = state["separate_verdicts"][candidate.candidate_id]
        rounds = 0
        current = candidate
        while not verdict.accepted and rounds < MAX_EDITOR_ROUNDS:
            evaluator_rejected += 1
            rounds += 1
            current, editor_raw = runtime.parsed_call(
                role="editor",
                mode=f"round-{rounds}",
                effort="low",
                system=(
                    "You repair a question without changing its "
                    "intended construct."
                ),
                prompt=build_editor_prompt(
                    current,
                    verdict,
                    state["topic"],
                    state["target_rank"],
                    rounds,
                ),
                schema=PracticeQuestionCandidate,
                tokens=state["token_calls"],
            )
            evaluation, evaluator_raw = evaluate(
                runtime,
                [current],
                state["topic"],
                state["target_rank"],
                mode=f"post-editor-{rounds}",
                tokens=state["token_calls"],
            )
            verdict = evaluation.verdicts[0]
            raw_calls.append(
                {
                    "round": rounds,
                    "editor": editor_raw,
                    "evaluator": evaluator_raw,
                }
            )
        rounds_used.append(rounds)
        if verdict.accepted:
            live.append(current)
        else:
            rejected.append(
                {
                    "candidate": current.model_dump(),
                    "stage": "editor-limit",
                    "violations": verdict.violations,
                }
            )
    return {
        "live": live,
        "rejected": rejected,
        "editor_raw": raw_calls,
        "editor_rounds": rounds_used,
        "evaluator_rejected": evaluator_rejected,
    }
