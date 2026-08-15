from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra.message_broker import ai_usage_tracker
from node.practiceEvalGraph.PronunciationNode.pronunciation_reference_helper import (
    build_pronunciation_reference,
)


def _fake_response(text: str = "corrected reference") -> SimpleNamespace:
    return SimpleNamespace(
        content=text,
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "input_token_details": {}},
    )


def test_claude_branch_records_usage_under_anthropic_provider() -> None:
    answer_id = "answer-claude-ref-test"
    ai_usage_tracker._usage_buffer.pop(answer_id, None)

    mock_llm = Mock()
    mock_llm.invoke.return_value = _fake_response()

    with patch(
        "node.practiceEvalGraph.PronunciationNode.pronunciation_reference_helper.ChatAnthropic",
        return_value=mock_llm,
    ):
        build_pronunciation_reference(
            "raw transcript", provider="claude", answer_id=answer_id
        )

    items = ai_usage_tracker.pop_usage(answer_id)
    assert len(items) == 1
    assert items[0].provider == "anthropic"
    assert items[0].model == "claude-sonnet-4-6"


def test_openai_branch_records_usage_under_openai_provider() -> None:
    answer_id = "answer-openai-ref-test"
    ai_usage_tracker._usage_buffer.pop(answer_id, None)

    mock_llm = Mock()
    mock_llm.invoke.return_value = _fake_response()

    with patch(
        "node.practiceEvalGraph.PronunciationNode.pronunciation_reference_helper.ChatOpenAI",
        return_value=mock_llm,
    ):
        build_pronunciation_reference(
            "raw transcript", provider="openai", answer_id=answer_id
        )

    items = ai_usage_tracker.pop_usage(answer_id)
    assert len(items) == 1
    assert items[0].provider == "openai"
    assert items[0].model == "gpt-5.4"
