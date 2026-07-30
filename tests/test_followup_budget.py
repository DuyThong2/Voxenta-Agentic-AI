from node.followUpDecisionGraph.FollowUpNode.followup_decision_node_config import (
    followup_decision_node,
)


def test_followup_stops_when_graded_budget_is_exhausted() -> None:
    result = followup_decision_node(
        {
            "current_turn": {"turn_order": 1, "transcript": "A full answer."},
            "remaining_graded_seconds": 0,
        }
    )

    assert result["followup_decision_result"] == {
        "should_continue": False,
        "next_prompt_text": None,
        "reason": "graded_budget_exhausted",
    }
