MAX_TURNS = 3

# Spoken by the avatar (via TTS, Phase 4) when a question's decision is should_continue=False
# and next_prompt_text is None -- followup_decision_node intentionally returns no prompt text in
# that case since there's nothing more to ask, but the avatar still needs something to say before
# RealtimeExamFlowService (WPF, Phase 5) advances to the next question_start.
CLOSING_REPLY = "Thank you for your answer. Let's move on."
EXAM_FAREWELL_TEXT = "That completes the speaking exam. Thank you for your effort today and good luck with your results!"
