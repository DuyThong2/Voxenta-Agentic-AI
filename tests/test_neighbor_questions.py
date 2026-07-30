from unittest.mock import Mock, patch

from vector.practice_question_selection import (
    NeighborQuestionRequest,
    neighbor_questions,
)


def test_neighbor_questions_queries_chroma_by_topic() -> None:
    collection = Mock()
    collection.query.return_value = {
        "ids": [["question-1", "question-2"]],
        "distances": [[0.1, 0.35]],
    }
    request = NeighborQuestionRequest(
        topic_name="Football and fan culture",
        criterion_code="GRAMMAR",
        rank_min=2,
        rank_max=4,
    )

    with patch(
        "vector.practice_question_selection.build_raw_collection",
        return_value=collection,
    ):
        response = neighbor_questions(request)

    collection.query.assert_called_once()
    assert [item.question_id for item in response.questions] == [
        "question-1",
        "question-2",
    ]
    assert [item.similarity for item in response.questions] == [0.9, 0.65]
