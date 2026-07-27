"""Phase 0 spike: compares two Voice Live input-transcription models
(default: mai-transcribe-1 vs gpt-4o-transcribe) on the SAME set of real
exam-turn audio files, to decide which one belongs in
AZURE_VOICELIVE_TRANSCRIPTION_MODEL (agents/.env) -- see
task/research/research-confidence-scoring.md section 1.1(a) for why this
question came up: gpt-4o-transcribe/gpt-4o-mini-transcribe are the only
Voice Live transcription models that return per-token logprobs (a real ASR
confidence signal), but public WER benchmarks put mai-transcribe-1 ahead,
especially on accented/non-native speech -- exactly this system's use case.
Public benchmarks aren't specific to Vietnamese-accented English, so this
script runs both models against this app's own real recorded exam audio
instead of trusting the benchmark numbers alone.

Standalone script, not wired into src/app.py -- run from the `agents`
directory:

    python spikes/compare_transcription_models_poc.py --urls-file spikes/sample-turn-urls.txt

--urls-file is a plain text file, one audio_url per line. Get a fresh list
of real turn URLs from the dev DB with:

    docker exec vox-db-postgres psql -U postgres -d vox -t -A \\
      -c "SELECT audio_url FROM exam_item_response_turns WHERE audio_url IS NOT NULL ORDER BY created_at DESC;"

Each URL is downloaded via the same infra.storage.audio_storage.download_from_s3
the real grading pipeline uses (agents/.env's AWS_* creds), so this exercises
the exact same download path as production, not a mocked one.

Requires `include=["item.input_audio_transcription.logprobs"]` in the session
config to get logprobs at all -- confirmed live that mai-transcribe-1 and
whisper-1 both reject this option outright (session.update fails with
max_config_attempts_exceeded), so mai-transcribe-1's confidence column is
always None here; that's expected, not a bug in this script.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import math
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SPIKES_DIR = Path(__file__).resolve().parent
AGENTS_DIR = SPIKES_DIR.parent
sys.path.insert(0, str(AGENTS_DIR / "src"))

from dotenv import load_dotenv

load_dotenv(dotenv_path=AGENTS_DIR / ".env")

from azure.ai.voicelive.aio import connect  # noqa: E402
from azure.ai.voicelive.models import (  # noqa: E402
    AudioInputTranscriptionOptions,
    InputAudioFormat,
    Modality,
    RequestSession,
    ServerEventType,
    ServerVad,
)
from azure.core.credentials import AzureKeyCredential  # noqa: E402
from infra.storage.audio_storage import download_from_s3  # noqa: E402

CHUNK_MS = 20
EXPECTED_SAMPLE_RATE = 16_000
CONNECT_TIMEOUT_SECONDS = 25.0
RESULTS_FILE = SPIKES_DIR / "poc-results.md"


@dataclass
class TranscribeResult:
    text: Optional[str]
    confidence: Optional[float]  # mean(exp(logprob)) across tokens, or None


async def _transcribe_once(wav_path: Path, transcription_model: str, want_logprobs: bool) -> TranscribeResult:
    include = ["item.input_audio_transcription.logprobs"] if want_logprobs else []
    try:
        async with connect(
            endpoint=os.environ["AZURE_VOICELIVE_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_VOICELIVE_API_KEY"]),
            model=os.getenv("AZURE_VOICELIVE_MODEL", "gpt-realtime"),
            api_version=os.getenv("AZURE_VOICELIVE_API_VERSION", "2026-04-10"),
        ) as conn:
            await conn.session.update(
                session=RequestSession(
                    modalities=[Modality.TEXT, Modality.AUDIO],
                    input_audio_format=InputAudioFormat.PCM16,
                    input_audio_sampling_rate=EXPECTED_SAMPLE_RATE,
                    input_audio_transcription=AudioInputTranscriptionOptions(model=transcription_model),
                    turn_detection=ServerVad(
                        threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500, create_response=False
                    ),
                    include=include,
                )
            )

            result = TranscribeResult(text=None, confidence=None)

            async def produce() -> None:
                with wave.open(str(wav_path), "rb") as wf:
                    if wf.getframerate() != EXPECTED_SAMPLE_RATE:
                        result.text = f"SKIP: {wf.getframerate()}Hz, expected {EXPECTED_SAMPLE_RATE}Hz"
                        return
                    frames_per_chunk = int(EXPECTED_SAMPLE_RATE * (CHUNK_MS / 1000))
                    while True:
                        frames = wf.readframes(frames_per_chunk)
                        if not frames:
                            break
                        await conn.input_audio_buffer.append(audio=base64.b64encode(frames).decode("ascii"))
                        await asyncio.sleep(CHUNK_MS / 1000)
                # Trailing silence so server-side VAD has something to detect speech_stopped in
                # -- same reasoning as voice_live_poc.py.
                silence_chunk = b"\x00" * (int(EXPECTED_SAMPLE_RATE * (CHUNK_MS / 1000)) * 2)
                for _ in range(1000 // CHUNK_MS):
                    await conn.input_audio_buffer.append(audio=base64.b64encode(silence_chunk).decode("ascii"))
                    await asyncio.sleep(CHUNK_MS / 1000)

            async def consume() -> None:
                async for event in conn:
                    if event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                        result.text = event.transcript
                        if event.logprobs:
                            probs = [math.exp(t.logprob) for t in event.logprobs]
                            result.confidence = round(sum(probs) / len(probs), 4)
                        return
                    if event.type == ServerEventType.ERROR:
                        result.text = f"ERROR: {event}"
                        return

            await asyncio.wait_for(asyncio.gather(produce(), consume()), timeout=CONNECT_TIMEOUT_SECONDS)
            return result
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is a comparison script,
        # one model/turn failing (timeout, session rejected, network) must not abort the run --
        # record it as a data point (a reliability difference IS the finding) and move on.
        return TranscribeResult(text=f"EXC: {type(exc).__name__}: {exc}", confidence=None)


async def run_comparison(urls: list[str], model_a: str, model_b: str) -> list[tuple[str, TranscribeResult, TranscribeResult]]:
    rows: list[tuple[str, TranscribeResult, TranscribeResult]] = []
    for i, url in enumerate(urls, start=1):
        try:
            local_path = Path(download_from_s3(url))
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(urls)}] DOWNLOAD FAILED {url}: {exc}")
            continue

        result_a = await _transcribe_once(local_path, model_a, want_logprobs=(model_a != "mai-transcribe-1"))
        result_b = await _transcribe_once(local_path, model_b, want_logprobs=(model_b != "mai-transcribe-1"))

        label = "/".join(url.split("/")[-2:])
        print(f"[{i}/{len(urls)}] {label}")
        print(f"    {model_a:<20}: {result_a.text!r} (confidence={result_a.confidence})")
        print(f"    {model_b:<20}: {result_b.text!r} (confidence={result_b.confidence})")
        rows.append((label, result_a, result_b))
    return rows


def append_results(model_a: str, model_b: str, rows: list[tuple[str, TranscribeResult, TranscribeResult]]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## compare_transcription_models_poc -- {model_a} vs {model_b} ({len(rows)} turns)\n\n")
        for label, result_a, result_b in rows:
            f.write(f"- `{label}`\n")
            f.write(f"  - {model_a}: {result_a.text!r} (confidence={result_a.confidence})\n")
            f.write(f"  - {model_b}: {result_b.text!r} (confidence={result_b.confidence})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urls-file", type=Path, required=True, help="Text file, one audio_url per line")
    parser.add_argument("--model-a", default="mai-transcribe-1")
    parser.add_argument("--model-b", default="gpt-4o-transcribe")
    args = parser.parse_args()

    urls = [line.strip() for line in args.urls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not urls:
        raise SystemExit(f"No URLs found in {args.urls_file}")

    rows = asyncio.run(run_comparison(urls, args.model_a, args.model_b))
    append_results(args.model_a, args.model_b, rows)
    print(f"\nResults appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
