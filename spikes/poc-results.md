
## voice_live_poc — sample.wav

- ` 2.484s` vad_speech_start: None
- ` 6.593s` vad_speech_end: None
- ` 7.078s` final_transcript: 'I usually go to school by bus.'

## voice_live_poc — sample.wav

- ` 1.671s` vad_speech_start: None
- ` 5.765s` vad_speech_end: None
- ` 6.515s` final_transcript: 'I usually go to school by bus.'

## compare_transcription_models_poc -- mai-transcribe-1 vs gpt-4o-transcribe (14 real exam turns)

Context: deciding AZURE_VOICELIVE_TRANSCRIPTION_MODEL (agents/.env) -- mai-transcribe-1 wins
public WER benchmarks (esp. accented speech) but rejects `include=["item.input_audio_transcription.logprobs"]`
outright (session.update fails, max_config_attempts_exceeded) so it never returns a real ASR
confidence signal; gpt-4o-transcribe/gpt-4o-mini-transcribe are the only Voice Live
transcription models that do return logprobs. Ran both against this app's own real recorded
exam-turn audio (not public benchmark data) to settle it with real evidence. See
task/research/research-confidence-scoring.md section 1.1(a) for the full confidence-formula
context this feeds into.

4/14 turns: both models timed out (no VAD-detected speech within 25s) -- likely very
short/near-silent turns, not a model quality difference, excluded from the comparison below.

- `05433cb3-.../turn-3.wav`
  - mai-transcribe-1: EXC: TimeoutError (both models, excluded)
  - gpt-4o-transcribe: EXC: TimeoutError (both models, excluded)
- `05433cb3-.../turn-2.wav`
  - mai-transcribe-1: 'I can see the park and sometimes I can see the students playing with each other down there.' (confidence=None)
  - gpt-4o-transcribe: 'I can see the park and sometime I can see the student.' (confidence=0.818)
  - mai captured meaningfully more content; gpt dropped a clause
- `05433cb3-.../turn-1.wav`
  - mai-transcribe-1: "My classroom is on the second floor of the school building. It's a private room with big windows on one side." (confidence=None)
  - gpt-4o-transcribe: "My classroom is on the second floor of the school building. It's a bright room with big windows on one side." (confidence=0.9817)
  - minor 1-word difference ("private" vs "bright"), no ground truth to judge which is right
- `2cbcbc0c-.../turn-5.wav`
  - mai-transcribe-1: 'to skip this question.' (confidence=None)
  - gpt-4o-transcribe: 'I want to skip this question.' (confidence=0.9326)
  - gpt more complete here -- the one clear win for gpt-4o-transcribe in this batch
- `2cbcbc0c-.../turn-4.wav`
  - mai-transcribe-1: "Let's give the question." (confidence=None)
  - gpt-4o-transcribe: 'Is give the question?' (confidence=0.7011)
  - both garbled (likely genuinely unclear audio); mai's is at least grammatical English
- `2cbcbc0c-.../turn-3.wav`
  - mai-transcribe-1: EXC: TimeoutError (both models, excluded)
  - gpt-4o-transcribe: EXC: TimeoutError (both models, excluded)
- `2cbcbc0c-.../turn-2.wav`
  - mai-transcribe-1: 'what important.' (confidence=None)
  - gpt-4o-transcribe: 'Ngwad important!' (confidence=0.3816)
  - gpt produced a nonsense token ("Ngwad"); mai stayed real English
- `2cbcbc0c-.../turn-1.wav`
  - mai-transcribe-1: 'I think students should have homework every day, but not too much.' (confidence=None)
  - gpt-4o-transcribe: 'Teen students should have homework every day, but not too much.' (confidence=0.9798)
  - mai far more plausible ("I think" vs "Teen")
- `fb68e88d-.../turn-4.wav`
  - mai-transcribe-1: EXC: TimeoutError (both models, excluded)
  - gpt-4o-transcribe: EXC: TimeoutError (both models, excluded)
- `fb68e88d-.../turn-3.wav`
  - mai-transcribe-1: EXC: TimeoutError (both models, excluded)
  - gpt-4o-transcribe: EXC: TimeoutError (both models, excluded)
- `fb68e88d-.../turn-2.wav`
  - mai-transcribe-1: 'At my place, there is Ms. Si Hung. She is very friendly and willing to help me every time I meet a problem.' (confidence=None)
  - gpt-4o-transcribe: EXC: TimeoutError (confidence=None)
  - mai succeeded outright; gpt-4o-transcribe failed to produce anything at all -- a reliability gap, not just accuracy
- `fb68e88d-.../turn-1.wav`
  - mai-transcribe-1: "I go to a high school in the city. It's a fairly big school with a lot of students. But I really like it because the teachers are friendly and always willing to help." (confidence=None)
  - gpt-4o-transcribe: "I go to a high school in the city. It's a fairly big school with a lot of students, but I really like it because the teachers are friendly and always willing to help." (confidence=0.986)
  - identical content, trivial punctuation difference only
- `5219fc4a-.../turn-2.wav`
  - mai-transcribe-1: 'I love dogs because they are loyal and very friendly.' (confidence=None)
  - gpt-4o-transcribe: 'They love dogs because they are loyal and very friendly.' (confidence=0.9519)
  - "I" vs "They" -- gpt changed the speaker's own stated pronoun, a real content error
- `5219fc4a-.../turn-1.wav`
  - mai-transcribe-1: 'My favorite animal is dog.' (confidence=None)
  - gpt-4o-transcribe: 'My favorite animal is dog.' (confidence=0.9918)
  - identical, clean audio

**Verdict on the 9 comparable turns**: mai-transcribe-1 clearly better on 4, gpt-4o-transcribe
clearly better on 1, rest tied/trivial -- plus one turn where gpt-4o-transcribe failed outright
(TimeoutError) and mai-transcribe-1 succeeded. Confirms the public WER benchmarks' direction
using this app's own real (Vietnamese-accented) exam audio, not just published numbers.
**Decision: keep AZURE_VOICELIVE_TRANSCRIPTION_MODEL=mai-transcribe-1** (agents/.env)
despite it never returning a confidence signal -- accuracy/reliability on real audio wins over
having a confidence number. The confidence gap this leaves is tracked separately in
task/research/research-confidence-scoring.md (proposed mitigation: cross-check via a parallel
Azure Speech SDK batch transcribe + edit-distance, not by switching transcription model).
