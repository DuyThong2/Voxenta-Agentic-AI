# Phase 0 spikes — setup notes

Throwaway PoC scripts for `docs/realtime-self-hosted-avatar-plan.md`'s Phase 0. Nothing here is
wired into the FastAPI app (`src/app.py`) — these are standalone scripts you run by hand to prove
feasibility before any of this gets productionized in Phase 3.

The self-hosted LivePortrait+MuseTalk avatar-rendering spike (`avatar_render_poc.py`,
`spikes/repos/`, `spikes/weights/`) has been removed: it was confirmed working end-to-end but
~18-25x slower than realtime on this hardware (RTX 2000 Ada, 8GB), and the project moved to an
audio-only "phone call" avatar (TTS only, no rendered video) while a hosted-avatar option (e.g.
Azure's realtime avatar) is evaluated for a future sprint. The `agents/pyproject.toml` dependencies
that existed only for that pipeline (mmcv/mmdet/mmpose/tensorflow/onnxruntime-gpu/diffusers/etc.)
were removed alongside it.

## Azure (for `voice_live_poc.py`) — DONE

This repo already uses Azure Speech for **archival** STT (`AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION`
in the root `.env`, consumed by `src/utils/speech_client.py`) — that part needs no new setup.

**Azure Voice Live (realtime STT+VAD) is a separate surface, accessed via the
`azure-ai-voicelive` SDK** (a real `pyproject.toml` dependency, added via
`uv add "azure-ai-voicelive[aiohttp]"` — installed `azure-ai-voicelive==1.2.0`). Already set up
in the root `.env`:
```
AZURE_VOICELIVE_ENDPOINT=https://<resource-name>.services.ai.azure.com/
AZURE_VOICELIVE_API_KEY=<key>
AZURE_VOICELIVE_MODEL=gpt-realtime-mini
AZURE_VOICELIVE_API_VERSION=2026-04-10
```
Confirmed against the installed SDK source directly (`.venv/Lib/site-packages/azure/ai/voicelive/`):
`connect()` accepts exactly these as `endpoint`/`credential`/`model`/`api_version` keywords, and
`api_version="2026-04-10"` is in fact the SDK's own current default. `gpt-realtime-mini` supports
input transcription models `whisper-1`/`gpt-4o-transcribe`/`gpt-4o-mini-transcribe`/
`gpt-4o-transcribe-diarize` (not `azure-speech`, which is for non-multimodal models/agents only)
— `voice_live_poc.py` defaults to `gpt-4o-mini-transcribe`, overridable via
`AZURE_VOICELIVE_TRANSCRIPTION_MODEL`.

`voice_live_poc.py`'s session is configured with `turn_detection.create_response=False` —
intentional: we only want Voice Live's STT+VAD here, not its own conversational response, since
the existing LangGraph follow-up decision logic decides what the avatar says next.

## GPU

`nvidia-smi` confirms a real NVIDIA GPU on this machine: RTX 2000 Ada Generation, ~8GB VRAM,
driver 595.71, CUDA 13.2 supported. `agents/pyproject.toml` pins `torch==2.3.0`/
`torchvision==0.18.0`/`torchaudio==2.3.0` against the `pytorch-cu118` index — this is no longer
driven by the (now-removed) avatar pipeline, just a known-working CUDA build for `ultralytics`/YOLO
proctoring. Re-verify with:
```powershell
cd D:\semester9\agents
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```
Must print a version ending in `+cu118`, `True`, and `NVIDIA RTX 2000 Ada Generation Laptop GPU`.

## Running

**Use `uv run`, not a bare `python`/`pip install`** — `azure-ai-voicelive` lives in
`agents/.venv` (added via `uv add`), not in any global Python install. If you run
`python spikes/voice_live_poc.py` directly with a global interpreter, the env-var check will
look fine but it'll then fail with `ModuleNotFoundError: azure.ai.voicelive`.

```bash
cd agents

# Already-correct-format test WAVs exist in data/ (16kHz/16-bit/mono, confirmed via `wave`):
#   data/sample.wav       (3.22s)
#   data/sampleError.wav  (4.74s)
#   data/sampleError2.wav (6.04s)
# data/note.txt has the ffmpeg command used to make sample.wav from a .m4a recording, in case
# you want to add more.
uv run python spikes/voice_live_poc.py --wav data/sample.wav
```
