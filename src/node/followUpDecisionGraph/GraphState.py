from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from node.state_models import QuestionContext


class FollowUpGraphState(TypedDict, total=False):
    answer_id: str
    exam_attempt_id: str
    # Attempt-scoped, not question-scoped like the rest of this state -- deliberately "borrowed"
    # onto the same archive_graph checkpointer (task/realtime-exam-flow-review.md), keyed by
    # exam_attempt_id as its own thread_id instead of answer_id. Written by
    # archive_store.set_current_answer_id right when _handle_question_start fires (before any
    # turn/decision exists for that question), read back by archive_store.get_current_answer_id
    # so a client that lost all local state (app fully closed, not just a WS reconnect) can ask
    # "which question was I on" without depending on Kafka's answer-turns-recorded topic, which
    # only fires after a turn completes and would otherwise be blind to a question that was
    # started but never got that far. LastValue (no reducer) -- each question_start simply
    # overwrites the previous value.
    current_answer_id: Optional[str]
    candidate_id: Optional[str]
    audio_ref: str
    paper_item_id: Optional[str]
    # question/language/prompt_text (below) are also, in archive_graph's
    # checkpoint specifically, the durable "question snapshot" persisted
    # once by archive_store.persist_question_snapshot at question_start --
    # read back by RealtimeExamSession.create_from_archive so a `resume` can
    # rebuild a full session from nothing but answer_id, with no question
    # data resent by the client.
    question: Optional[QuestionContext]
    language: str
    audio_path: str
    turn_order: int
    prompt_text: Optional[str]
    # Doubles as the decision graph's per-turn input AND (in archive_graph's
    # checkpoint specifically) the durably-persisted pending follow-up
    # prompt -- written by turn_publisher.publish_turn_if_new after each
    # turn's decision, read back by archive_store.get_resume_state /
    # RealtimeExamSession._apply_resume_state on reconnect. LastValue (no
    # reducer) -- each write simply overwrites the previous one, which is
    # exactly the "latest pending prompt" semantics wanted.
    active_prompt_text: Optional[str]
    current_turn: Dict[str, Any]
    turns: Annotated[List[Dict[str, Any]], add]
    published_turn_orders: Annotated[List[int], add]
    # [{"turn_order": int, "reason": str}, ...] -- turns archived via POST
    # /turns/archive never carry decision_reason (that's only known in-memory
    # by RealtimeExamSession after decide_next_step runs), so it's persisted
    # here separately by turn_publisher.publish_turn_if_new and merged back
    # onto the archived turns at resume time by archive_store.get_resume_state.
    decision_reasons: Annotated[List[Dict[str, Any]], add]
    # [{"turn_order": int, "text": str}, ...] -- the live Voice-Live (gpt-4o-mini-transcribe)
    # transcript captured during the exam itself (see attempt_connection.py's
    # [realtime_transcript] logging), persisted durably here so eval-time scoring can prefer
    # it over re-transcribing the archived audio via the Azure Speech SDK -- Voice-Live
    # handles code-switched Vietnamese noticeably better (confirmed: it correctly transcribes
    # words the Speech SDK sometimes garbles into nonsense English, e.g. "banh MI" instead of
    # "bánh mì"). See archive_store.persist_realtime_transcript / exam_consumer.py.
    realtime_transcripts: Annotated[List[Dict[str, Any]], add]
    signals: Dict[str, Any]
    edge_case_handled: bool
    decision: Dict[str, Any]
    repeat_recovery_edge_case_handled: bool
    repeat_recovery_decision: Dict[str, Any]
    repeat_recovery_error: Optional[str]
    followup_decision_result: Dict[str, Any]
    followup_decision_error: Optional[str]
    status: Literal["idle", "processing", "completed", "error"]
    error: Optional[str]
