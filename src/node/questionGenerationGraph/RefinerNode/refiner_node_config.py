from node.questionGenerationGraph.constants import REFINER_BATCH_SIZE
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
)
from node.questionGenerationGraph.RefinerNode.refiner_prompt import (
    build_refiner_prompt,
)
from schemas.question_generation import RefinedBatch


def refiner_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    refined = []
    raw_calls = list(state["editor_raw"])
    live = state["live"]
    for start in range(0, len(live), REFINER_BATCH_SIZE):
        chunk = live[start : start + REFINER_BATCH_SIZE]
        polished, raw = runtime.parsed_call(
            role="refiner",
            mode=f"batch-{len(chunk)}",
            effort="low",
            system=(
                "You perform light independent copy-editing on "
                "accepted questions."
            ),
            prompt=build_refiner_prompt(
                chunk,
                state["topic"],
                state["target_rank"],
            ),
            schema=RefinedBatch,
            tokens=state["token_calls"],
        )
        refined.extend(polished.candidates)
        raw_calls.append({"refiner": raw})
    return {
        "refined": refined,
        "editor_raw": raw_calls,
    }
