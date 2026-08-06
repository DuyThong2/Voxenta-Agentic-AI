import uuid

from node.questionGenerationGraph.CandidateFilterNode.candidate_filter_node_config import (
    rule_violations,
)
from node.questionGenerationGraph.constants import ALLOWED_SUB_ATTRIBUTES
from node.questionGenerationGraph.graphConfig import QuestionGenerationGraph
from node.questionGenerationGraph.question_generation_graph_helper import (
    question_record,
)
from node.questionGenerationGraph.question_generation_persistence import (
    index_question,
)
from schemas.question_generation import (
    GeneratedQuestion,
    PracticeQuestionCandidate,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionIndexRequest,
)

_graph: QuestionGenerationGraph | None = None


def generate_questions(
    request: QuestionGenerationRequest,
) -> QuestionGenerationResponse:
    graph = _question_graph()
    topic = (
        request.topic_name,
        request.interest_dimension,
        request.curriculum_group,
    )
    criterion = (
        request.target_criterion_code,
        request.target_sub_attribute,
    )
    # fast=True: đây là đường ONLINE (Java gọi khi học sinh đang chờ vào phiên).
    # pipeline.py (nghiên cứu) vẫn gọi graph.invoke mặc định fast=False nên số
    # liệu đã đo không đổi. Xem constants.FAST_* để biết cắt những gì.
    state = graph.invoke(
        topic,
        criterion,
        request.target_rank,
        fast=True,
        needed=request.count,
        band_count=request.band_count,
        band_ladder=request.band_ladder,
        exclude_question_ids=request.exclude_question_ids,
    )
    records = []
    for candidate in state["refined"]:
        if rule_violations(candidate, request.target_sub_attribute):
            continue
        records.append(
            question_record(
                question_id=str(uuid.uuid4()),
                topic=topic,
                candidate=candidate,
                topic_id=request.topic_id,
                band_count=request.band_count,
            )
        )
        if len(records) >= request.count:
            break

    if not records:
        salvaged = _salvage_one(state, request)
        if salvaged is not None:
            records.append(
                question_record(
                    question_id=str(uuid.uuid4()),
                    topic=topic,
                    candidate=salvaged,
                    topic_id=request.topic_id,
                    band_count=request.band_count,
                )
            )

    return QuestionGenerationResponse(
        questions=[GeneratedQuestion.model_validate(item) for item in records]
    )


def _salvage_one(state, request) -> PracticeQuestionCandidate | None:
    """Cuu lay DUNG 1 cau khi khong ung vien nao qua duoc cong.

    Vi sao phai co: tra ve 0 cau kem HTTP 200 la che do hong te nhat -- Java lap qua danh
    sach rong, khong loi, khong log, roi bao "chu de chua co cau luyen phu hop". Ca luong
    luyen tap dung lai ma khong tang nao bao gi. Tha dua ra mot cau hoi kha di con hon
    khong cho hoc sinh luyen duoc gi.

    Chi sua nhung gi sua duoc mot cach TAT DINH (sub-attribute lech taxonomy), khong dung
    them LLM: day la duong online, hoc sinh dang doi. Cau nao hong phan khong sua duoc bang
    tay (qua ngan/qua dai/khong phai tieng Anh) thi bo qua.
    """
    allowed = ALLOWED_SUB_ATTRIBUTES.get(request.target_criterion_code)
    if allowed is None:
        return None
    pool = list(state.get("refined") or [])
    pool += [
        candidate
        for candidate in (state.get("live") or [])
        if candidate not in pool
    ]
    pool += [
        candidate
        for candidate in (state.get("survivors") or [])
        if candidate not in pool
    ]
    for candidate in pool:
        # Thu nguyen trang truoc -- ung vien co the da hop le, chi bi evaluator che ve chat
        # luong. Khong sua cai dang dung.
        if not rule_violations(candidate, request.target_sub_attribute):
            return candidate
        # Chi con vi pham sub-attribute thi vá tất định: uu tien dung gia tri Java yeu cau,
        # khong yeu cau gi thi de trong (null = "luyen tieu chi nay noi chung").
        patched = candidate.model_copy(deep=True)
        patched.target_sub_attribute = (
            request.target_sub_attribute
            if request.target_sub_attribute in allowed
            else (None if None in allowed else None)
        )
        if not rule_violations(patched, request.target_sub_attribute):
            return patched
    return None


def index_generated_question(request: QuestionIndexRequest) -> None:
    question = request.question.model_dump()
    index_question(_question_graph().runtime, question)


def _question_graph() -> QuestionGenerationGraph:
    global _graph
    if _graph is None:
        _graph = QuestionGenerationGraph()
    return _graph
