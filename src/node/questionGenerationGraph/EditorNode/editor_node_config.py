from concurrent.futures import ThreadPoolExecutor

from node.questionGenerationGraph.constants import (
    FAST_EDITOR_ROUNDS,
    FAST_EVALUATOR_EFFORT,
    MAX_EDITOR_ROUNDS,
)
from node.questionGenerationGraph.EditorNode.editor_prompt import (
    build_editor_prompt,
)
from node.questionGenerationGraph.EvaluatorNode.evaluator_node_config import (
    evaluate,
    ladder_for,
)
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
)
from schemas.question_generation import PracticeQuestionCandidate


def _repair_one(
    candidate: PracticeQuestionCandidate,
    verdict,
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
    *,
    max_rounds: int,
    effort: str,
) -> dict:
    """Vòng sửa của ĐÚNG 1 ứng viên -- không đụng state dùng chung ngoài
    token_calls (list.append là atomic dưới GIL), nên chạy song song được."""
    rounds = 0
    current = candidate
    raw_calls = []
    evaluator_rejected = 0
    while not verdict.accepted and rounds < max_rounds:
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
            effort=effort,
            band_ladder=ladder_for(state),
        )
        verdict = evaluation.verdicts[0]
        raw_calls.append(
            {
                "round": rounds,
                "editor": editor_raw,
                "evaluator": evaluator_raw,
            }
        )
    return {
        "candidate": current,
        "verdict": verdict,
        "rounds": rounds,
        "raw_calls": raw_calls,
        "evaluator_rejected": evaluator_rejected,
    }


def editor_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    fast = bool(state.get("fast"))
    max_rounds = FAST_EDITOR_ROUNDS if fast else MAX_EDITOR_ROUNDS
    effort = FAST_EVALUATOR_EFFORT if fast else "high"
    survivors = state["survivors"]
    verdicts = state["separate_verdicts"]

    # Ứng viên đã ĐẠT ngay từ lượt chấm đầu thì không cần sửa gì -- tách ra
    # trước để (a) fast mode có thể bỏ qua hẳn phần sửa nếu đã đủ số câu cần,
    # (b) chỉ những câu thật sự phải sửa mới tốn call.
    passed = [c for c in survivors if verdicts[c.candidate_id].accepted]
    failed = [c for c in survivors if not verdicts[c.candidate_id].accepted]

    # Đủ câu rồi thì dừng -- đường online chỉ cần `needed` câu, sửa thêm chỉ
    # tốn thời gian học sinh đang phải chờ.
    if fast and len(passed) >= max(1, state.get("needed", 1)):
        return {
            "live": passed,
            "rejected": list(state["rejected"]),
            "editor_raw": [],
            "editor_rounds": [0] * len(passed),
            "evaluator_rejected": 0,
        }

    results = []
    if failed:
        # Mỗi nhánh sửa là 2 call TUẦN TỰ (editor -> chấm lại), nhưng các nhánh
        # của những ứng viên khác nhau hoàn toàn độc lập -> chạy song song để
        # tổng thời gian bằng nhánh chậm nhất thay vì tổng mọi nhánh.
        with ThreadPoolExecutor(max_workers=len(failed)) as pool:
            results = list(
                pool.map(
                    lambda candidate: _repair_one(
                        candidate,
                        verdicts[candidate.candidate_id],
                        state,
                        runtime,
                        max_rounds=max_rounds,
                        effort=effort,
                    ),
                    failed,
                )
            )

    live = list(passed)
    rejected = list(state["rejected"])
    raw_calls = []
    rounds_used = [0] * len(passed)
    evaluator_rejected = 0
    for result in results:
        rounds_used.append(result["rounds"])
        raw_calls.extend(result["raw_calls"])
        evaluator_rejected += result["evaluator_rejected"]
        if result["verdict"].accepted:
            live.append(result["candidate"])
        else:
            rejected.append(
                {
                    "candidate": result["candidate"].model_dump(),
                    "stage": "editor-limit",
                    "violations": result["verdict"].violations,
                }
            )
    return {
        "live": live,
        "rejected": rejected,
        "editor_raw": raw_calls,
        "editor_rounds": rounds_used,
        "evaluator_rejected": evaluator_rejected,
    }
