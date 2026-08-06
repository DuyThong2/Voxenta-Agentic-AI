from langgraph.graph import END, START, StateGraph

from node.questionGenerationGraph.CandidateFilterNode import (
    candidate_filter_node,
)
from node.questionGenerationGraph.DrafterNode import drafter_node
from node.questionGenerationGraph.EditorNode import editor_node
from node.questionGenerationGraph.EvaluatorNode import evaluator_node
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.RefinerNode import refiner_node
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
)


class QuestionGenerationGraph:
    def __init__(
        self,
        runtime: QuestionGenerationRuntime | None = None,
    ) -> None:
        self.runtime = runtime or QuestionGenerationRuntime()
        graph = StateGraph(QuestionGenerationState)
        graph.add_node(
            "drafter",
            lambda state: drafter_node(state, self.runtime),
        )
        graph.add_node(
            "candidate_filter",
            lambda state: candidate_filter_node(state, self.runtime),
        )
        graph.add_node(
            "evaluator",
            lambda state: evaluator_node(state, self.runtime),
        )
        graph.add_node(
            "editor",
            lambda state: editor_node(state, self.runtime),
        )
        graph.add_node(
            "refiner",
            lambda state: refiner_node(state, self.runtime),
        )
        graph.add_edge(START, "drafter")
        graph.add_edge("drafter", "candidate_filter")
        graph.add_edge("candidate_filter", "evaluator")
        graph.add_edge("evaluator", "editor")
        graph.add_edge("editor", "refiner")
        graph.add_edge("refiner", END)
        self.compiled = graph.compile()

    def invoke(
        self,
        topic: tuple[str, str, str],
        criterion: tuple[str, str | None],
        target_rank: int,
        *,
        fast: bool = False,
        needed: int = 1,
        band_count: int = 6,
        band_ladder=None,
        exclude_question_ids=None,
        target_tense: str | None = None,
    ) -> QuestionGenerationState:
        """fast=True: đường online (học sinh đang chờ) -- xem constants.FAST_*.
        Mặc định False để pipeline nghiên cứu giữ nguyên phương pháp đã đo."""
        return self.compiled.invoke(
            {
                "topic": topic,
                "criterion": criterion,
                "target_rank": target_rank,
                "target_tense": target_tense,
                "fast": fast,
                "needed": needed,
                "band_count": band_count,
                "band_ladder": list(band_ladder or []),
                "exclude_question_ids": set(exclude_question_ids or ()),
                "token_calls": [],
                "rejected": [],
                "filter_reasons": set(),
                "cosines": [],
                "editor_rounds": [],
                "evaluator_rejected": 0,
                "comparison_total": 0,
                "comparison_different": 0,
            }
        )
