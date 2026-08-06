from node.questionGenerationGraph.constants import (
    FAST_EVALUATOR_EFFORT,
    build_band_ladder,
)
from node.questionGenerationGraph.EvaluatorNode.evaluator_prompt import (
    build_evaluator_prompt,
)
from node.questionGenerationGraph.GraphState import QuestionGenerationState
from node.questionGenerationGraph.question_generation_graph_helper import (
    QuestionGenerationRuntime,
    verdict_signature,
)
from schemas.question_generation import (
    CandidateVerdict,
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
    effort: str = "high",
    band_ladder: str | None = None,
) -> tuple[EvaluationBatch, dict]:
    return runtime.parsed_call(
        role="evaluator",
        mode=mode,
        effort=effort,
        system=(
            "You are a strict independent evaluator. "
            "The band ladder at the beginning of the user message is authoritative."
        ),
        prompt=build_evaluator_prompt(candidates, topic, target_rank, band_ladder),
        schema=EvaluationBatch,
        tokens=tokens,
    )


def ladder_for(state: QuestionGenerationState) -> str:
    """Ladder mo ta thang bac dung tu du lieu Java gui xuong; rong thi lui ve hang so mac dinh."""
    return build_band_ladder(
        state.get("band_ladder"),
        state.get("band_count", 6),
    )


def evaluator_node(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    if state.get("fast"):
        return _fast_evaluator(state, runtime)

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
            band_ladder=ladder_for(state),
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
            band_ladder=ladder_for(state),
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


def _fast_evaluator(
    state: QuestionGenerationState,
    runtime: QuestionGenerationRuntime,
) -> dict:
    """Đường ONLINE: nhận thẳng mọi ứng viên đã qua candidate_filter, KHÔNG gọi LLM.

    Vì sao bỏ hẳn lượt chấm ở đây: học sinh đang ngồi chờ, mà lượt này tốn 1 call và kéo
    theo editor (thêm 2 call mỗi ứng viên bị chê). Bỏ nó đưa đường online từ 2-4 lượt gọi
    xuống còn ĐÚNG 1 (drafter).

    Cổng chất lượng còn lại vẫn đủ chặt cho đường online:
      - schema structured-output ép đủ 6 trường evaluation_guide, ngân sách thời gian, vstep_part
      - rule_violations: độ dài 6..80 từ, >=90% ký tự Latin, sub-attribute đúng taxonomy
      - kiểm trùng lặp bằng embedding trong candidate_filter
      - service._salvage_one vá nốt trường hợp không ứng viên nào sạch

    Pipeline nghiên cứu (fast=False) VẪN chấm đầy đủ separate + grouped, không đụng tới.
    """
    verdicts = {
        candidate.candidate_id: CandidateVerdict(
            candidate_id=candidate.candidate_id,
            accepted=True,
            violations=[],
        )
        for candidate in state["survivors"]
    }
    return {
        "separate_verdicts": verdicts,
        "comparison_total": 0,
        "comparison_different": 0,
        "evaluator_raw": {"fast": "skipped-online"},
    }
