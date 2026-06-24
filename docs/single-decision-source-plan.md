# Single decision source: Tavus decides, this repo only archives + relays "done" via tool call

Companion to `DesktopApp/VoxOralExam/docs/single-decision-source-plan.md` (WPF side) — read that
for the full picture; this is the `agents`-side half. As before, this repo can't see the WPF
source, so everything below is the verified contract WPF will call.

✅ Implemented and import-verified. **Update:** WPF now keeps **one Tavus conversation for the
whole exam** instead of one per question (recreating per question caused a UX flicker) — it sends
`conversation.overwrite_llm_context` between questions instead. Consequence for this repo: the
`messages` array `/v1/chat/completions` receives can now contain previous questions' turns too.
Fixed in `mappers/chat_completion_mapper.py` — `_extract_question_context`/`extract_answer_id`
now read the *most recent* system message (not the first), and `build_followup_state_from_messages`
scopes turn-counting to whatever comes after that message, so turn_order/turns never bleed across
questions. Verified with a multi-question synthetic `messages` list.

## Why this changes

Today `/evaluate/turn` (`followup_controller.py`) makes its own follow-up decision using Azure
STT on the uploaded audio — independently from `/v1/chat/completions` (`tavus_controller.py`),
which makes a *different* live decision using Tavus's own STT. The two can disagree. Decided:
**`/v1/chat/completions` becomes the only decision-maker.** WPF stops calling `/evaluate/turn` for
decisions and instead calls a new endpoint, `/turns/archive`, purely to hand over each turn's
S3 audio link for record-keeping — no decision logic runs there anymore.

The two endpoints still need to agree on the same question/answer, since `/turns/archive` gets
`answer_id` directly from WPF but `/v1/chat/completions` only ever sees whatever WPF put in
Tavus's `conversational_context` (relayed back as a system message). WPF will additionally embed
`<answer_id>{guid}</answer_id>` next to the existing `<question_context>` marker — add a small
`_extract_answer_id(messages)` helper to `mappers/chat_completion_mapper.py` (same regex-marker
approach as `_extract_question_context`, just a plain string instead of JSON).

## New: `POST /turns/archive`

New file, e.g. `controller/archive_controller.py`. Same request shape as today's `/evaluate/turn`
(`audio_ref`, `answer_id`, `turn_order`, `prompt_text`, `language`, `question`) — copy that
route's `Form(...)` signature. Differs in what it does:

1. Download the audio from S3 (`infra/storage/audio_storage.download_from_s3`, same as today).
2. Transcribe it (`utils/speech_client.transcribe`, same Azure STT call as today's
   `transcribe_turn_node`) — **this repo keeps producing the higher-quality archived transcript**,
   that part of the design was never the problem; only the *decision* duplication was.
3. Append the resulting turn to a Postgres-checkpointed list keyed by `thread_id=answer_id` —
   reuse the same checkpointer `app.state` already holds for `build_followup_graph`. Build a new,
   smaller graph with no decision node:

   ```python
   def build_archive_graph(checkpointer):
       g = StateGraph(FollowUpGraphState)
       g.add_node("transcribe_turn", transcribe_turn_node)        # reuse as-is from graphConfig.py
       g.add_node("append_turn", append_turn_node)                 # new, see below
       g.add_edge(START, "transcribe_turn")
       g.add_edge("transcribe_turn", "append_turn")
       g.add_edge("append_turn", END)
       return g.compile(checkpointer=checkpointer)

   def append_turn_node(state):
       current_turn = state["current_turn"]
       return {**state, "status": "archived", "turns": [current_turn]}  # same Annotated[list, add] accumulation pattern prepare_turn_signals_node already uses
   ```

4. Respond with a small ack, no decision fields: `{"turn_order": ..., "status": "archived"}`.
   WPF does not branch on this response — don't design it as if it matters for control flow.

Build this graph at startup the same way `app.py` already builds `app.state.graph`/
`app.state.followup_graph`/`app.state.text_followup_graph`; call it `app.state.archive_graph`.

## Changed: `POST /v1/chat/completions` publishes the Kafka event now

`followup_decision_node` itself doesn't change. What changes is in `tavus_controller.py`'s
`chat_completions` handler: when the decision's `should_continue` is `False`, before replying,
look up the **archived** turns for this `answer_id` (extracted via the new
`_extract_answer_id`) from the same Postgres checkpointer `archive_graph` uses — e.g.
`archive_graph.get_state({"configurable": {"thread_id": answer_id}}).values.get("turns", [])` —
and publish `AnswerTurnsRecordedEvent` from *that* (it has real `audio_url`s and Azure-quality
transcripts; the message-history-derived `turns` `build_followup_state_from_messages` produces do
not). This replaces the Kafka-publish call that currently lives in `followup_controller.py`'s
`evaluate_turn` — move it here, don't keep publishing from both places.

⚠️ **Timing edge case, not resolved:** if WPF's `/turns/archive` call for the *last* turn hasn't
completed by the time Tavus's live decision resolves `should_continue=False`, that last turn
would be missing from the published event (the archive simply hasn't landed yet — these are two
independent async calls racing each other, same root cause as the original two-decisions problem,
just narrowed to "is the archive in time" instead of "which decision is right"). No mitigation
designed; a short bounded wait/retry before publishing is the likely fix if this shows up in
testing, not a redesign.

## Changed: relay "question done" to the client via a tool call

Tavus's tool-calling docs say a Custom LLM's response can include `tool_calls` (OpenAI format),
and "Tavus does not execute tool calls on the backend... use event listeners in your frontend" —
i.e. it just relays the tool call to the WebView2/Daily client as an app-message, which is
exactly the "done" signal WPF needs instead of guessing from spoken text. When
`should_continue=False`, include a tool call alongside the closing reply content:

```python
# dtos/response/chat_completion.py — add to ChatCompletionMessage / ChatCompletionChoice
class ToolCallFunction(_pydantic-or-plain-BaseModel_):
    name: str
    arguments: str = "{}"

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction

# ChatMessage (or a response-specific variant) needs an optional tool_calls: Optional[List[ToolCall]] = None
```

```python
# mappers/chat_completion_mapper.py
def build_end_question_tool_call() -> ToolCall:
    return ToolCall(id=f"call_{uuid.uuid4().hex}", function=ToolCallFunction(name="end_question"))
```

Wire this into `build_chat_completion_response`/`_stream_chunks` so that when
`resolve_reply_content` would return the closing text, the response's assistant message also
carries `tool_calls=[build_end_question_tool_call()]`.

⚠️ **Exact request schema Tavus needs to register this tool isn't pinned down yet** — Tavus's
tool-calling docs describe tools being declared under the Persona's `layers.llm.tools` config
(a Persona-level setting, configured once via the Tavus dashboard/API, not per-request) —
someone needs to register an `end_question` tool (no required arguments) on the Persona this
exam uses, separately from this repo's code change. Flag this to whoever manages the Tavus
Persona config; it's not something `agents` code can do by itself.

## Verification

- Unit: call the new `chat_completions` handler with a synthetic `messages` history whose answer
  is clearly sufficient (`should_continue` should resolve `False`), and confirm the response
  includes a `tool_calls` entry with `function.name == "end_question"`.
- Integration: call `/turns/archive` twice with the same `answer_id` (`turn_order` 1 then 2), then
  call `/v1/chat/completions` for that same `answer_id` driving `should_continue=False`, and
  confirm the published `AnswerTurnsRecordedEvent` contains both archived turns with real
  `audio_url`/`transcript`/`word_count` — not the thin message-derived versions.
