from infra.message_broker.publishers.exam_publisher import practice_event_body


class _Event:
    def model_dump(self, *, by_alias: bool) -> dict:
        assert by_alias
        return {
            "eventType": "ExamAttemptEvaluationCompleted",
            "examAttemptId": "session-1",
            "answerId": "response-2",
            "questionId": "question-3",
            "payload": {},
        }


def test_practice_completion_uses_session_partition_and_ids() -> None:
    body, partition_key = practice_event_body(
        _Event(),
        "PracticeAttemptEvaluationCompleted",
    )

    assert partition_key == "session-1"
    assert body["practiceSessionId"] == "session-1"
    assert body["practiceResponseId"] == "response-2"
    assert body["practiceQuestionId"] == "question-3"
