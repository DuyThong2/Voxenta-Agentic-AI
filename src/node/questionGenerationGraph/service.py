import uuid

from node.questionGenerationGraph.CandidateFilterNode.candidate_filter_node_config import (
    rule_violations,
)
from node.questionGenerationGraph.graphConfig import QuestionGenerationGraph
from node.questionGenerationGraph.question_generation_graph_helper import (
    question_record,
)
from node.questionGenerationGraph.question_generation_persistence import (
    index_question,
)
from schemas.question_generation import (
    GeneratedQuestion,
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
    )
    records = []
    for candidate in state["refined"]:
        if rule_violations(candidate):
            continue
        records.append(
            question_record(
                question_id=str(uuid.uuid4()),
                topic=topic,
                candidate=candidate,
                topic_id=request.topic_id,
            )
        )
        if len(records) >= request.count:
            break
    return QuestionGenerationResponse(
        questions=[GeneratedQuestion.model_validate(item) for item in records]
    )


def index_generated_question(request: QuestionIndexRequest) -> None:
    question = request.question.model_dump()
    index_question(_question_graph().runtime, question)


def _question_graph() -> QuestionGenerationGraph:
    global _graph
    if _graph is None:
        _graph = QuestionGenerationGraph()
    return _graph
