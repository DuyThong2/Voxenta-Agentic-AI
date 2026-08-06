from node.onlineQuestionGenerationGraph.SelectorEditorNode.selector_editor_prompt import (
    build_selector_editor_prompt,
)
from node.questionGenerationGraph.constants import build_band_ladder
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
)
from schemas.question_generation import PracticeQuestionCandidate


def selector_editor_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    """Doc ca N ung vien, chon 1, chinh no, chot question_type + thoi luong. MOT luot LLM.

    Thay cho bo ba evaluator/editor/refiner o duong online cu -- bo ba do co ve chay nhung
    thuc te KHONG cham vao cau nao: _fast_evaluator gat het (accepted=True), ma cua vao editor
    lai la `while not verdict.accepted`, con refiner thi `if state.get("fast")` bo qua. Ket qua
    la service lay UNG VIEN DAU TIEN qua duoc rule_violations -- khong ai chon co chu dich,
    khong ai chinh sua, va question_type/thoi luong chi do drafter doan roi bi kep mu bang so
    hoc trong _clamp_to_type_range.

    Node nay la cho DUY NHAT co mot con mat doc noi dung cau hoi roi phan: cau nay hop bac
    chua, no thuc su la DESCRIPTION hay SHORT_ANSWER, 45 giay cho no co hop ly khong. Bo kep
    theo dai van chay sau (validator cua PracticeQuestionCandidate) nhung gio dung dung vai
    tro luoi an toan, khong con la nguoi quyet dinh.
    """
    survivors = state.get("survivors") or []
    if not survivors:
        return {"live": [], "editor_raw": []}

    # Mot ung vien thi khong con gi de CHON -- nhung van goi de CHINH va chot loai/thoi luong,
    # vi do moi la phan chinh cua node nay.
    chosen, raw = runtime.parsed_call(
        role="selector_editor",
        mode=f"of-{len(survivors)}",
        effort="low",
        system=(
            "You choose and finish one English speaking-practice question "
            "for a specific learner band."
        ),
        prompt=build_selector_editor_prompt(
            survivors,
            state["topic"],
            state["criterion"],
            state["target_rank"],
            build_band_ladder(state.get("band_ladder"), state.get("band_count", 6)),
        ),
        schema=PracticeQuestionCandidate,
        tokens=state["token_calls"],
    )

    return {
        "live": [chosen],
        "editor_raw": [{"selector_editor": raw}],
    }
