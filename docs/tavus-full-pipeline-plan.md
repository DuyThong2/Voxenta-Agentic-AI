# Tavus Full Pipeline custom-LLM endpoint — plan / ground truth

This doc is for whoever (codex or otherwise) is working in the `agents` repo only, without
visibility into the `DesktopApp/VoxOralExam` (WPF) or `vox` (Java) repos. Treat it as the
authoritative description of what the WPF client actually does and expects, verified against the
WPF source directly (not guessed).

## Architecture decision that matters here: one Tavus conversation per exam question

WPF creates a **new Tavus conversation for every exam question** (not one conversation for the
whole exam). It calls `POST /v2/conversations` once per question, with that question's
`conversational_context` embedded, joins it, runs until the question is "done", then calls
`DELETE /v2/conversations/{id}` and moves to the next question with a brand-new conversation.

**Why this matters for the code in this repo:** it means a single Tavus conversation — and
therefore every `POST /v1/chat/completions` call Tavus makes for that conversation — is always
about exactly **one** question. There is no "move to the next question" concept that needs to
live in this repo's graph state. `build_text_followup_graph`'s current design (no checkpointer,
state rebuilt fresh from `messages` on every call) is correct and doesn't need to change for
this — keep it that way.

## What WPF sends as `conversational_context` (the system message)

At conversation creation, WPF builds a string like this and Tavus relays it back as a `system`
role message in `payload.messages` on every subsequent `/v1/chat/completions` call for that
conversation (this relaying behavior is what `_extract_question_context` in
`mappers/chat_completion_mapper.py` already depends on and is confirmed working):

```
You are conducting an English speaking exam for question {N} of {M}.
Exam instructions: {question.InstructionText}
<question_context>{"question_text": "...", "question_type": "...", "difficulty_level": "...", "duration_seconds": ..., "min_response_seconds": ..., "max_response_seconds": ..., "evaluation_guide": "..."}</question_context>
```

The `<question_context>` JSON is snake_case and matches `node.state_models.speaking_input.QuestionContext` field-for-field. `question_type`/`difficulty_level` fall back to generic
defaults (`"speaking"`/`"medium"`) on the WPF side if the mock question bank doesn't have a more
specific value — don't assume every conversation will have a precise type/difficulty.

## Minor, optional: `CLOSING_REPLY` wording

WPF used to plan on detecting "this question is done" by string-matching whatever the Tavus
replica says out loud. That's no longer how WPF decides this — it now calls `/evaluate/turn`
directly per turn (see below) and uses the structured `should_continue`/`reached_max_turns`
fields from the response, not the spoken text. So `CLOSING_REPLY` in
`mappers/chat_completion_mapper.py` (`"Thank you, that's all for this question."`, returned by
`resolve_reply_content` when `should_continue=False`) is no longer load-bearing for WPF's control
flow — only cosmetic (it's still what the avatar actually says to the student when Tavus's own
live decision agrees the question is done). Fine to leave as-is; tightening the wording so a
student couldn't accidentally trigger it is still good practice but not urgent anymore.

## What actually matters now: keep the Kafka event aligned with Java's real `AnswerTurn` entity

WPF calls `POST /evaluate/turn` (already built, `controller/followup_controller.py`) for **every
turn**, with the turn's S3 `audio_ref` included — not just for the follow-up decision text, but
because this is the real persistence path: when `should_continue` goes `False`,
`evaluate_turn` publishes `AnswerTurnsRecordedEvent` (topic `answer-turns-recorded`) and that's
meant to be the thing a Java Kafka consumer eventually reads to persist `AnswerTurn` rows. Java
has no consumer for this topic yet, but the payload shape this repo produces **should already be
a close-to-1:1 mirror of Java's actual `AnswerTurn` entity** so that whenever that consumer gets
written, it doesn't need extra mapping logic. Java's entity (don't redefine this independently —
treat it as the source of truth) is:

```java
// vox: domain/model/examAttempt/AnswerTurn.java
UUID id; UUID answerId; Integer turnOrder; TurnType turnType; // enum: MAIN, FOLLOWUP
String promptText; String audioUrl; String transcript;
Integer durationSeconds; Integer wordCount; OffsetDateTime answeredAt;
```

Current `AnswerTurnPayload` (`events/answer_turns_recorded.py`) already names every field to
match (`answer_id`→`answerId`, `turn_order`→`turnOrder`, `turn_type`→`turnType`, `prompt_text`→
`promptText`, `audio_url`→`audioUrl`, `transcript`, `duration_seconds`→`durationSeconds`,
`word_count`→`wordCount`, `answered_at`→`answeredAt` — camelCase via `_CamelMessage`'s
`alias_generator`). `turn_type` already comes through as the literal string `"MAIN"`/`"FOLLOWUP"`
(`transcribe_turn_node` in `graphConfig.py`), which matches Java's enum constant names exactly —
good, don't touch that.

**The actual gap:** in `transcribe_turn_node` (`node/followUpDecisionGraph/graphConfig.py`,
lines ~23-33), `duration_seconds` and `answered_at` are hardcoded `None` and **nothing anywhere
else in the graph ever sets them** (confirmed — grepped the whole `followUpDecisionGraph`
package). Java's `AnswerTurn.durationSeconds`/`answeredAt` are real columns meant to hold actual
values, not permanent nulls — every turn that ever reaches Java through this event will be
missing them otherwise. Fix both at the point `audio_path` is already available (it's already
downloaded from S3 for transcription at that point, no new I/O needed):

- `duration_seconds`: read the WAV file's own duration directly — it's a 16kHz mono PCM16 file
  WPF produced, so Python's stdlib `wave` module is enough, no new dependency:
  ```python
  import wave
  with wave.open(audio_path, "rb") as f:
      duration_seconds = round(f.getnframes() / f.getframerate())
  ```
- `answered_at`: set to the current UTC time at the point this node runs (closest available
  proxy — the precise client-side mic timestamp isn't transmitted today):
  ```python
  from datetime import datetime, timezone
  answered_at = datetime.now(timezone.utc).isoformat()
  ```

Set both in the `current_turn` dict instead of the hardcoded `None`s. Nothing else in this repo
needs to change for the Java alignment — the rest of the shape already matches.

## Verification

- Unit-level: invoke `transcribe_turn_node` with a real WAV `audio_path` and confirm
  `current_turn["duration_seconds"]` matches the file's actual length and
  `current_turn["answered_at"]` is a real ISO-8601 UTC timestamp, not `None`.
- Integration: drive a full `/evaluate/turn` call to `should_continue=false`, capture the
  published `AnswerTurnsRecordedEvent` (e.g. log it before/instead of actually publishing to
  Kafka in a local test), and confirm every turn in `payload.turns` has non-null
  `durationSeconds`/`answeredAt` alongside the already-correct `answerId`/`turnOrder`/`turnType`/
  `promptText`/`audioUrl`/`transcript`/`wordCount`.
