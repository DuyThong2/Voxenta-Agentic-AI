"""Do thi cham bai LUYEN TAP -- chep tu evalGraph, bo dung mot node.

Khac evalGraph o DUY NHAT mot cho: khong co AzureScoreScaleNode.

Vi sao bo duoc: node do anh xa diem HundredMark cua Azure (0-100) vao dai [min, max] cua tung
tieu chi rubric. Luyen tap tu V13 cham thang 0-100 co dinh -- dung thang goc Azure tra ve -- nen
phep anh xa thanh x -> x. Java gui rubric_min_score=0, rubric_max_score=100 va chi MOT bac (bac
hoc sinh tu chon), targetBandOnly=true.

Vi sao TACH HAN thay vi dung chung node: sua duong luyen khong duoc phep cham vao duong thi. Cai
gia la ~2900 dong chep doi va se troi lech -- danh doi da chon co y thuc. Duong lui neu thay
phien hon loi: giu file nay, xoa cac thu muc node da chep, cho import lai tu node.evalGraph.

MergeScoresNode PHAI giu: no la diem hop luu cua hai nhanh song song, khong lien quan rubric.
"""

from langgraph.graph import END, START, StateGraph

from node.practiceEvalGraph.GraphState import GraphState
from node.practiceEvalGraph.LanguageQualityEvalNode.language_quality_eval_node_config import (
    language_quality_eval_node,
)
from node.practiceEvalGraph.PronunciationNode.pronunciation_eval_node_config import (
    pronunciation_eval_node,
)
from node.practiceEvalGraph.AnswerLengthNode.answer_length_analysis_node_config import (
    answer_length_analysis_node,
)
from node.practiceEvalGraph.MergeScoresNode.merge_scores_node_config import merge_scores_node
from node.practiceEvalGraph.StartNode.start_node_config import start_node
from node.practiceEvalGraph.ValidityNode.validity_node_config import validity_node


def route_after_validity(state: GraphState) -> list[str] | str:
    """Route to END if validity rejects. Otherwise fan out to pronunciation_eval AND
    answer_length_analysis for complete answers.

    Multi-turn fragments only need per-audio pronunciation. Text validity, answer length,
    and language quality run once later against the merged complete answer.
    """
    validity = state.get("validity")
    if validity and getattr(validity, "action", None) == "reject_or_zero":
        return "end"
    metadata = state.get("metadata") or {}
    if metadata.get("validity_scope") == "turn_fragment":
        return "pronunciation_eval"
    return ["pronunciation_eval", "answer_length_analysis"]


def route_after_answer_length(state: GraphState) -> list[str] | str:
    """Language quality needs answer_length_metrics for its criterion-specific caps.

    If answer_length_analysis failed, skip the LLM calls because their result would be
    discarded at merge time.
    """
    metadata = state.get("metadata") or {}
    if metadata.get("answer_length_error"):
        return "merge_scores"
    return "language_quality_eval"


def route_on_error(state: GraphState) -> str:
    if state.get("status") == "error":
        return "end"
    return "continue"


def build_practice_graph(checkpointer=None):
    g = StateGraph(GraphState)

    g.add_node("start", start_node)
    g.add_node("strict_validity_check", validity_node)
    g.add_node("pronunciation_eval", pronunciation_eval_node)
    g.add_node("answer_length_analysis", answer_length_analysis_node)
    g.add_node("language_quality_eval", language_quality_eval_node)
    g.add_node("merge_scores", merge_scores_node)

    # No CorrectionNode: it fed an LLM-rewritten transcript into validity/pronunciation
    # scoring, but the rewrite was frequently wrong (over-corrected disfluencies, mangled
    # code-switched wording) -- both scoring and display now go straight off Azure's own
    # transcription (speech_client.transcribe(), which already handles code-switched
    # Vietnamese via auto-detect + language tagging), no LLM rewrite step in between.
    g.add_edge(START, "start")
    g.add_conditional_edges(
        "start",
        route_on_error,
        {
            "end": END,
            "continue": "strict_validity_check",
        },
    )

    # Fan-out #1: pronunciation_eval (Azure Speech SDK call) and answer_length_analysis
    # (word/sentence counting + optional LLM length judgment) run concurrently -- neither
    # depends on the other's output.
    g.add_conditional_edges(
        "strict_validity_check",
        route_after_validity,
        {
            "end": END,
            "pronunciation_eval": "pronunciation_eval",
            "answer_length_analysis": "answer_length_analysis",
        },
    )

    # Once answer_length_analysis has produced the score caps, one combined node makes three
    # parallel O-C-O requests. Every request returns all three language criteria, while
    # consensus/confidence remains criterion-specific.
    g.add_conditional_edges(
        "answer_length_analysis",
        route_after_answer_length,
        {
            "merge_scores": "merge_scores",
            "language_quality_eval": "language_quality_eval",
        },
    )

    # Azure always returns HundredMark. Scale its criterion scores to each
    # RubricCriterion range before the final fan-in.
    # Thang cua Azure GIU NGUYEN 0-100, khong con buoc anh xa nao o giua.
    g.add_edge("pronunciation_eval", "merge_scores")
    g.add_edge("language_quality_eval", "merge_scores")
    g.add_edge("merge_scores", END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)

    return g.compile()
