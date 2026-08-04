"""Đồ thị sinh câu hỏi cho đường ONLINE -- học sinh đang ngồi chờ vào phiên.

Tách hẳn khỏi questionGenerationGraph (đường nghiên cứu, chạy batch offline) thay vì dùng
chung một đồ thị rồi rẽ nhánh bằng cờ `fast`. Lý do là bài học rút từ chính cờ đó:

Nhìn đồ thị cũ thấy 5 node nối nhau đàng hoàng -- drafter, candidate_filter, evaluator,
editor, refiner -- nên ai đọc cũng tưởng cả 5 đều chạy. Sự thật ở đường online là 3 trong 5
KHÔNG chạm vào câu nào: _fast_evaluator gật hết (accepted=True), cửa vào editor lại là
`while not verdict.accepted` nên không vòng nào chạy, còn refiner thì `if state.get("fast")`
bỏ qua. Muốn biết điều đó phải mở ba file khác -- cấu trúc nói dối về hành vi.

Ở đây đồ thị nói đúng thứ nó làm: ba node, cả ba đều chạy, đúng hai lượt gọi LLM.

    START → drafter → candidate_filter → selector_editor → END
              1 lượt        0 lượt            1 lượt

Đường nghiên cứu giữ nguyên đồ thị cũ và phương pháp cũ -- số liệu trong seed-output/ đo bằng
nó, đổi hành vi là làm hỏng cơ sở so sánh.
"""

from langgraph.graph import END, START, StateGraph

from node.onlineQuestionGenerationGraph.SelectorEditorNode.selector_editor_node_config import (
    selector_editor_node,
)
from node.questionGenerationGraph.CandidateFilterNode.candidate_filter_node_config import (
    candidate_filter_node,
)
from node.questionGenerationGraph.DrafterNode.drafter_node_config import drafter_node
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
)


class OnlineQuestionGenerationGraph:
    def __init__(self, runtime: QuestionGenerationRuntime | None = None) -> None:
        self.runtime = runtime or QuestionGenerationRuntime()
        graph = StateGraph(QuestionGenerationState)
        # drafter và candidate_filter DÙNG LẠI của đồ thị nghiên cứu, không chép: chúng là
        # cùng một việc (sinh ứng viên, lọc theo luật + chống trùng bằng embedding). Chép ra
        # bản thứ hai là mở đường cho hai bản trôi lệch nhau.
        graph.add_node("drafter", lambda state: drafter_node(state, self.runtime))
        graph.add_node(
            "candidate_filter",
            lambda state: candidate_filter_node(state, self.runtime),
        )
        graph.add_node(
            "selector_editor",
            lambda state: selector_editor_node(state, self.runtime),
        )
        graph.add_edge(START, "drafter")
        graph.add_edge("drafter", "candidate_filter")
        graph.add_edge("candidate_filter", "selector_editor")
        graph.add_edge("selector_editor", END)
        self.compiled = graph.compile()

    def invoke(
        self,
        topic: tuple[str, str, str],
        criterion: tuple[str, str | None],
        target_rank: int,
        *,
        band_count: int = 6,
        band_ladder=None,
    ) -> QuestionGenerationState:
        return self.compiled.invoke(
            {
                "topic": topic,
                "criterion": criterion,
                "target_rank": target_rank,
                "band_count": band_count,
                "band_ladder": list(band_ladder or []),
                "token_calls": [],
                "rejected": [],
                "filter_reasons": set(),
                "cosines": [],
                "editor_rounds": [],
                "evaluator_rejected": 0,
                "comparison_total": 0,
                "comparison_different": 0,
                "editor_raw": [],
            }
        )


_graph: OnlineQuestionGenerationGraph | None = None


def get_online_graph() -> OnlineQuestionGenerationGraph:
    global _graph
    if _graph is None:
        _graph = OnlineQuestionGenerationGraph()
    return _graph
