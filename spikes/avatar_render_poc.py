"""Phase 0 spike: run LivePortrait (motion/expression) then MuseTalk (lip-sync) on a driving
audio clip + reference photo, and benchmark end-to-end latency/FPS.

Standalone script, not wired into src/app.py — see spikes/README.md for setup (Hugging Face
weights download, GPU requirement). Run from the `agents` directory:

    python spikes/avatar_render_poc.py --audio <driving_audio.wav> --photo <reference_photo.jpg>

The actual LivePortrait/MuseTalk inference calls are intentionally NotImplementedError stubs
(see `_run_liveportrait` / `_run_musetalk` below) — the exact Python entrypoint depends on which
version of each repo you end up cloning alongside the downloaded weights (their APIs aren't
stable across versions). Everything else here (timing per stage, peak memory, output video
writing, results logging) is already correct and shouldn't need to change.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("avatar_render_poc")

SPIKES_DIR = Path(__file__).resolve().parent
AGENTS_DIR = SPIKES_DIR.parent  # .env lives at agents/.env, not one level above that
RESULTS_FILE = SPIKES_DIR / "poc-results.md"

DEFAULT_LIVEPORTRAIT_WEIGHTS_DIR = SPIKES_DIR / "weights" / "liveportrait"
DEFAULT_MUSETALK_WEIGHTS_DIR = SPIKES_DIR / "weights" / "musetalk"

LIVEPORTRAIT_REPO_DIR = SPIKES_DIR / "repos" / "LivePortrait"
MUSETALK_REPO_DIR = SPIKES_DIR / "repos" / "MuseTalk"
OUTPUTS_DIR = SPIKES_DIR / "outputs"

# LivePortrait's pipeline expects weights at <repo>/pretrained_weights/... — a directory
# junction at LIVEPORTRAIT_REPO_DIR/pretrained_weights -> DEFAULT_LIVEPORTRAIT_WEIGHTS_DIR (the
# hf-downloaded copy) satisfies this without duplicating multi-GB files. Likewise MuseTalk
# expects <repo>/models/... via a junction to DEFAULT_MUSETALK_WEIGHTS_DIR. Both junctions are
# created once by hand (see spikes/README.md) — this script just assumes they already resolve.


@dataclass
class StageTiming:
    stage: str
    seconds: float
    peak_memory_mb: Optional[float] = None


@dataclass
class AvatarRenderConfig:
    liveportrait_weights_dir: Path
    musetalk_weights_dir: Path


def load_config() -> AvatarRenderConfig:
    load_dotenv(dotenv_path=AGENTS_DIR / ".env")
    liveportrait_dir = Path(os.getenv("LIVEPORTRAIT_WEIGHTS_DIR", str(DEFAULT_LIVEPORTRAIT_WEIGHTS_DIR)))
    musetalk_dir = Path(os.getenv("MUSETALK_WEIGHTS_DIR", str(DEFAULT_MUSETALK_WEIGHTS_DIR)))

    missing = [d for d in (liveportrait_dir, musetalk_dir) if not d.exists()]
    if missing:
        raise SystemExit(
            "Missing weights directory(ies): "
            + ", ".join(str(d) for d in missing)
            + ". See agents/spikes/README.md section 2 for the huggingface-cli download commands. "
            "(Not downloaded yet is expected — fill these in once you've pulled the weights.)"
        )

    return AvatarRenderConfig(liveportrait_weights_dir=liveportrait_dir, musetalk_weights_dir=musetalk_dir)


def _peak_memory_mb() -> Optional[float]:
    """Best-effort peak RSS in MB. Returns None if psutil isn't available rather than failing
    the whole spike over a missing optional dependency."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _torch_lib_dir() -> Optional[Path]:
    """torch (cu118 build) bundles cudart64_110.dll/cublas64_11.dll/cudnn64_8.dll/etc. under
    <venv>/Lib/site-packages/torch/lib, which onnxruntime-gpu's CUDA execution provider depends
    on but doesn't bundle or locate itself — adding this to PATH/child env is harmless and was
    the fix for the original "LoadLibrary error 126" (missing dependency). It does NOT fix
    onnxruntime's CUDA provider end to end, though: even with this directory present, it still
    fails with `WinError 1114: DLL initialization routine failed` (the provider DLL's own init
    code crashing, most likely an exact cuDNN patch-version mismatch between what onnxruntime-gpu
    was built against and what torch bundles) — confirmed by directly loading the DLL via
    ctypes. onnxruntime permanently falls back to CPU for face-detection; this is an accepted,
    low-impact state (see spikes/README.md's GPU section), not a bug still being chased."""
    torch_lib = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "torch" / "lib"
    return torch_lib if torch_lib.is_dir() else None


def _run_subprocess(args: list[str], cwd: Path, stage: str) -> subprocess.CompletedProcess:
    # Both repos print Unicode (emoji progress labels, etc.) — on Windows, a subprocess with its
    # stdout/stderr redirected to a pipe falls back to the legacy ANSI codepage (cp1252) unless
    # told otherwise, which can't encode those characters and crashes the child entirely. Force
    # UTF-8 in the child's own I/O regardless of the parent console's codepage.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    torch_lib = _torch_lib_dir()
    if torch_lib is not None:
        child_env["PATH"] = f"{torch_lib}{os.pathsep}{child_env.get('PATH', '')}"
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env
    )
    if result.returncode != 0:
        logger.error("[%s] stdout (tail):\n%s", stage, result.stdout[-4000:])
        logger.error("[%s] stderr (tail):\n%s", stage, result.stderr[-4000:])
        raise RuntimeError(f"{stage} failed (exit {result.returncode}) — see logged stdout/stderr above")
    return result


def _run_liveportrait(config: AvatarRenderConfig, photo_path: Path, audio_path: Path) -> tuple[Path, StageTiming]:
    """Drive `photo_path` with motion/expression cues and return a path to the resulting
    (silent) motion video, plus timing for this stage.

    `audio_path` is intentionally unused: the real KwaiVGI/LivePortrait repo animates a still
    photo from a *driving video or motion template* (`-d`/`--driving`), not from raw audio —
    confirmed by reading `src/config/argument_config.py` directly. Lip-sync to `audio_path` is
    MuseTalk's job, in `_run_musetalk` below. This PoC uses one of LivePortrait's own bundled
    example driving clips (`assets/examples/driving/d0.mp4`) as the driving source.
    """
    del audio_path

    driving_video = LIVEPORTRAIT_REPO_DIR / "assets" / "examples" / "driving" / "d0.mp4"
    output_dir = OUTPUTS_DIR / "liveportrait"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    _run_subprocess(
        [
            sys.executable, "inference.py",
            "-s", str(photo_path.resolve()),
            "-d", str(driving_video.resolve()),
            "-o", str(output_dir.resolve()),
        ],
        cwd=LIVEPORTRAIT_REPO_DIR,
        stage="liveportrait",
    )
    elapsed = time.monotonic() - start

    # LivePortraitPipeline.execute names its output "{source_stem}--{driving_stem}.mp4" (with a
    # "_with_audio" suffix only when the driving clip itself carries audio, which d0.mp4 may or
    # may not — check both before falling back to a glob).
    candidates = [
        output_dir / f"{photo_path.stem}--{driving_video.stem}_with_audio.mp4",
        output_dir / f"{photo_path.stem}--{driving_video.stem}.mp4",
    ]
    output_video = next((p for p in candidates if p.exists()), None)
    if output_video is None:
        found = sorted(output_dir.glob(f"{photo_path.stem}--{driving_video.stem}*.mp4"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise RuntimeError(f"LivePortrait reported success but no output video found in {output_dir}")
        output_video = found[-1]

    return output_video, StageTiming(stage="liveportrait", seconds=elapsed, peak_memory_mb=_peak_memory_mb())


def _run_musetalk(config: AvatarRenderConfig, motion_video_path: Path, audio_path: Path) -> tuple[Path, StageTiming]:
    """Lip-sync `motion_video_path` to `audio_path` and return a path to the final output video,
    plus timing for this stage.

    Drives the real TMElyralab/MuseTalk repo via its `scripts/inference.py` CLI (run as
    `-m scripts.inference` so its `from musetalk...` absolute imports resolve — running the file
    directly would only put `scripts/` on sys.path, not the repo root). Weights are reached via
    the `<repo>/models -> spikes/weights/musetalk` junction (see spikes/README.md) — `v15`
    paths (`models/musetalkV15/...`) are used since that's the version actually downloaded.
    """
    output_dir = OUTPUTS_DIR / "musetalk"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_id = "vox_poc"
    inference_config_path = output_dir / "inference_config.yaml"
    inference_config_path.write_text(
        f"{task_id}:\n"
        f"  video_path: {motion_video_path.resolve().as_posix()}\n"
        f"  audio_path: {audio_path.resolve().as_posix()}\n",
        encoding="utf-8",
    )

    start = time.monotonic()
    _run_subprocess(
        [
            sys.executable, "-m", "scripts.inference",
            "--inference_config", str(inference_config_path.resolve()),
            "--unet_config", "./models/musetalkV15/musetalk.json",
            "--unet_model_path", "./models/musetalkV15/unet.pth",
            "--whisper_dir", "./models/whisper",
            "--vae_type", "sd-vae",
            "--use_float16",
            "--version", "v15",
            "--result_dir", str(output_dir.resolve()),
        ],
        cwd=MUSETALK_REPO_DIR,
        stage="musetalk",
    )
    elapsed = time.monotonic() - start

    output_basename = f"{motion_video_path.stem}_{audio_path.stem}"
    output_video = output_dir / "v15" / f"{output_basename}.mp4"
    if not output_video.exists():
        found = sorted(output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise RuntimeError(f"MuseTalk reported success but no output video found in {output_dir}")
        output_video = found[-1]

    return output_video, StageTiming(stage="musetalk", seconds=elapsed, peak_memory_mb=_peak_memory_mb())


def run_poc(audio_path: Path, photo_path: Path) -> list[StageTiming]:
    config = load_config()
    timings: list[StageTiming] = []
    overall_start = time.monotonic()

    stage_start = time.monotonic()
    motion_video_path, liveportrait_timing = _run_liveportrait(config, photo_path, audio_path)
    timings.append(liveportrait_timing)
    logger.info("LivePortrait stage: %.3fs", time.monotonic() - stage_start)

    stage_start = time.monotonic()
    output_video_path, musetalk_timing = _run_musetalk(config, motion_video_path, audio_path)
    timings.append(musetalk_timing)
    logger.info("MuseTalk stage: %.3fs", time.monotonic() - stage_start)

    logger.info(
        "End-to-end: %.3fs, output at %s", time.monotonic() - overall_start, output_video_path
    )
    return timings


def append_results(audio_path: Path, photo_path: Path, timings: list[StageTiming]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## avatar_render_poc — audio={audio_path.name} photo={photo_path.name}\n\n")
        for timing in timings:
            mem = f", peak {timing.peak_memory_mb:.0f}MB" if timing.peak_memory_mb else ""
            f.write(f"- {timing.stage}: {timing.seconds:.3f}s{mem}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Driving audio clip (WAV)")
    parser.add_argument("--photo", type=Path, required=True, help="Reference avatar photo")
    args = parser.parse_args()

    timings = run_poc(args.audio, args.photo)
    append_results(args.audio, args.photo, timings)


if __name__ == "__main__":
    main()
