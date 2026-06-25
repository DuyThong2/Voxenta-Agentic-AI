# Phase 0 PoC results

Appended to automatically by `voice_live_poc.py` and `avatar_render_poc.py`. Used to ground the
Phase 0 go/no-go gate in `docs/realtime-self-hosted-avatar-plan.md` (e.g. the ~15-20fps strawman
threshold for LivePortrait+MuseTalk, and the "<500ms after actual silence" strawman for Voice
Live VAD latency — both are placeholders until real numbers land here).

No runs yet — both scripts currently stop at a `NotImplementedError` until Azure Voice Live
credentials and the LivePortrait/MuseTalk weights+repos are set up (see `spikes/README.md`).

## voice_live_poc — sample.wav

- ` 2.140s` vad_speech_start: None
- ` 5.453s` vad_speech_end: None
- ` 6.172s` partial_transcript: 'I'
- ` 6.172s` partial_transcript: ' usually'
- ` 6.172s` partial_transcript: ' go'
- ` 6.172s` partial_transcript: ' to'
- ` 6.187s` partial_transcript: ' school'
- ` 6.187s` partial_transcript: ' by'
- ` 6.187s` partial_transcript: ' bus'
- ` 6.187s` partial_transcript: '.'
- ` 6.312s` final_transcript: 'I usually go to school by bus.'

## voice_live_poc — sample.wav

- ` 2.141s` vad_speech_start: None
- `3601.204s` error: "{'event_id': 'event_1we6LMKKTlOOZlShZoFGyt', 'type': 'error', 'error': {'message': 'Your session hit the maximum duration of 60 minutes.', 'type': 'invalid_request_error', 'code': 'session_expired', 'param': None, 'event_id': None}}"

## voice_live_poc — sample.wav

- ` 2.141s` vad_speech_start: None
- ` 5.453s` vad_speech_end: None
- ` 6.297s` partial_transcript: 'I'
- ` 6.297s` partial_transcript: ' usually'
- ` 6.297s` partial_transcript: ' go'
- ` 6.297s` partial_transcript: ' to'
- ` 6.297s` partial_transcript: ' school'
- ` 6.297s` partial_transcript: ' by'
- ` 6.297s` partial_transcript: ' bus'
- ` 6.297s` partial_transcript: '.'
- ` 6.578s` final_transcript: 'I usually go to school by bus.'

## voice_live_poc — sample.wav

- ` 2.594s` vad_speech_start: None
- ` 5.781s` vad_speech_end: None
- ` 6.547s` partial_transcript: 'I'
- ` 6.547s` partial_transcript: ' usually'
- ` 6.547s` partial_transcript: ' go'
- ` 6.547s` partial_transcript: ' to'
- ` 6.547s` partial_transcript: ' school'
- ` 6.563s` partial_transcript: ' by'
- ` 6.563s` partial_transcript: ' bus'
- ` 6.563s` partial_transcript: '.'
- ` 6.766s` final_transcript: 'I usually go to school by bus.'

## avatar_render_poc — audio=sample.wav photo=image.png

- liveportrait: 19.907s, peak 22MB
- musetalk: 196.469s, peak 22MB

## avatar_render_poc — audio=sample.wav photo=image.png

- liveportrait: 21.734s, peak 22MB
- musetalk: 57.797s, peak 22MB

## avatar_render_poc — audio=sample.wav photo=image.png

- liveportrait: 28.516s, peak 23MB
- musetalk: 165.532s, peak 23MB

## avatar_render_poc — audio=sample.wav photo=image.png

- liveportrait: 29.750s, peak 23MB
- musetalk: 124.094s, peak 23MB
