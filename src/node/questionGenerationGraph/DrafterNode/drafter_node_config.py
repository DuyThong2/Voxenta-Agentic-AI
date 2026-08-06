from node.questionGenerationGraph.DrafterNode.drafter_prompt import (
    build_drafter_prompt,
)
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
)
from schemas.question_generation import DraftBatch


def drafter_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    draft, raw = runtime.parsed_call(
        role="drafter",
        mode="batch",
        effort="low",
        system=(
            "You draft inclusive English speaking-practice questions. "
            "Return structured data only."
        ),
        prompt=build_drafter_prompt(
            state["topic"],
            state["criterion"],
            state["target_rank"],
            # Cùng thang bậc mà EvaluatorNode nhận. Trước đây chỉ nút CHẤM biết thang, còn
            # nút VIẾT chỉ nhận một con số trần -- bên viết không hiểu đích, bên chấm thì hiểu.
            state.get("band_ladder"),
            state.get("band_count", 6),
        ),
        schema=DraftBatch,
        tokens=state["token_calls"],
    )
    return {
        "candidates": draft.candidates,
        "drafter_raw": raw,
    }
