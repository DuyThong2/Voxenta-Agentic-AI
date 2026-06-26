"""Production avatar renderer (Phase 4 of docs/realtime-self-hosted-avatar-plan.md).

Two-stage pipeline per exam attempt:

1. LivePortrait (subprocess, reusing spikes/avatar_render_poc.py's proven CLI wiring almost
   verbatim) drives the student-facing avatar reference photo with a bundled motion template into
   one silent "motion video". Runs ONCE per exam attempt (not per utterance) -- the photo and
   driving template are constant for the whole attempt, so there is nothing question-specific to
   re-render here.
2. MuseTalk (in-process, NOT subprocess) lip-syncs that one motion video to each turn's TTS audio.
   This is promoted from spikes/avatar_render_poc.py's one-shot `scripts.inference` CLI call --
   which reloaded every model checkpoint from disk on every single utterance -- to the *cached*
   Avatar-based path MuseTalk itself ships for realtime/conversational use (see its own
   scripts/realtime_inference.py): landmark/face-parsing/latent extraction for the motion video's
   frames happens ONCE per attempt (ensure_avatar_ready), and each utterance's render_utterance
   call only runs the much cheaper UNet+VAE decode step. Phase 0's PoC benchmark (79.5s-216s per
   utterance) never exercised this cached path, so those numbers significantly understate what
   this renderer can do -- Phase 4's own timing (logged per call below) is the real number.

Resolves Open Question 10b (in-process vs. subprocess for Phase 4) in favor of in-process for
MuseTalk: the dependency-isolation reason that originally motivated a separate venv
(spikes/.venv-avatar, now deleted) no longer applies now that agents/.venv is the single, unified
dependency manifest for this whole repo (mmcv/mmdet/mmpose/torch are all resolved there already).
LivePortrait stays subprocess-based since it only runs once per attempt -- not worth porting.
"""

import asyncio
import copy
import glob
import logging
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

AGENTS_DIR = Path(__file__).resolve().parent.parent.parent
SPIKES_DIR = AGENTS_DIR / "spikes"
LIVEPORTRAIT_REPO_DIR = SPIKES_DIR / "repos" / "LivePortrait"
MUSETALK_REPO_DIR = SPIKES_DIR / "repos" / "MuseTalk"

# Durable, on-disk cache -- separate from spikes/outputs (which is throwaway PoC output) since
# this is the production renderer's working state and survives a process restart on purpose
# (re-preparing an avatar's MuseTalk landmarks/latents is the expensive part this exists to avoid).
AVATAR_VAR_DIR = AGENTS_DIR / "var" / "avatar"
LIVEPORTRAIT_OUTPUT_DIR = AVATAR_VAR_DIR / "liveportrait"
MUSETALK_RESULT_DIR = AVATAR_VAR_DIR / "musetalk"

DEFAULT_REFERENCE_PHOTO = Path(os.getenv("AVATAR_REFERENCE_PHOTO", str(AGENTS_DIR / "data" / "image.png")))
MUSETALK_VERSION = "v15"
MUSETALK_FPS = 25


# --------------------------------------------------------------------------------------
# Stage 1: LivePortrait (subprocess, once per attempt) -- reused from avatar_render_poc.py
# --------------------------------------------------------------------------------------

def _torch_lib_dir() -> Optional[Path]:
    """See spikes/avatar_render_poc.py's identical helper: torch's cu118 build bundles the CUDA
    DLLs LivePortrait's onnxruntime dependency needs to even attempt loading its CUDA provider
    (it permanently falls back to CPU regardless -- accepted, not chased further, see
    agents/pyproject.toml's onnxruntime comment)."""
    torch_lib = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "torch" / "lib"
    return torch_lib if torch_lib.is_dir() else None


def _run_subprocess(args: List[str], cwd: Path, stage: str) -> subprocess.CompletedProcess:
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    torch_lib = _torch_lib_dir()
    if torch_lib is not None:
        child_env["PATH"] = f"{torch_lib}{os.pathsep}{child_env.get('PATH', '')}"
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env
    )
    if result.returncode != 0:
        logger.error("[avatar_renderer:%s] stdout (tail):\n%s", stage, result.stdout[-4000:])
        logger.error("[avatar_renderer:%s] stderr (tail):\n%s", stage, result.stderr[-4000:])
        raise RuntimeError(f"{stage} failed (exit {result.returncode}) -- see logged stdout/stderr above")
    return result


def _run_liveportrait_sync(avatar_id: str, photo_path: Path) -> Path:
    """Idempotent: returns the cached motion video for avatar_id if it already exists (e.g. a
    `resume` after a process restart, or a second question in the same attempt) instead of
    re-rendering."""
    output_dir = LIVEPORTRAIT_OUTPUT_DIR / avatar_id
    candidates = [
        output_dir / f"{photo_path.stem}--d0_with_audio.mp4",
        output_dir / f"{photo_path.stem}--d0.mp4",
    ]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is not None:
        logger.info("[avatar_renderer] LivePortrait motion video already cached for avatar_id=%s", avatar_id)
        return existing

    driving_video = LIVEPORTRAIT_REPO_DIR / "assets" / "examples" / "driving" / "d0.mp4"
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
    logger.info("[avatar_renderer] LivePortrait motion video rendered in %.1fs (avatar_id=%s)", time.monotonic() - start, avatar_id)

    output_video = next((p for p in candidates if p.exists()), None)
    if output_video is None:
        found = sorted(output_dir.glob(f"{photo_path.stem}--d0*.mp4"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise RuntimeError(f"LivePortrait reported success but no output video found in {output_dir}")
        output_video = found[-1]
    return output_video


# --------------------------------------------------------------------------------------
# Stage 2: MuseTalk (in-process, models loaded once, cached per-attempt landmark/latent prep)
# --------------------------------------------------------------------------------------

_musetalk_models: Optional[dict] = None

# os.chdir is process-wide, not per-thread -- a real risk if some unrelated concurrent request
# handler reads a relative path while this is in effect. Audited the rest of this app (utils/env.py,
# utils/jsonl_logger.py, etc.) and confirmed everything else already uses __file__-anchored
# absolute paths, so the actual exposure is narrow; this lock at least serializes MuseTalk/
# LivePortrait calls against each other (on top of gpu_scheduler's GPU semaphore, which already
# limits real concurrency to 1 by default) and the cwd is restored immediately after each call
# rather than left changed for the renderer's lifetime. Forking MuseTalk to take absolute paths
# (or running it as a fully separate worker process) would remove this caveat entirely -- a
# reasonable Phase 6 hardening candidate, not done here.
# RLock (not Lock): _prepare_avatar_sync enters the guard and then calls
# _load_musetalk_models_sync, which enters its own nested guard on the same thread -- a plain
# Lock would deadlock there.
_musetalk_cwd_lock = threading.RLock()


class _MuseTalkCwdGuard:
    """Context manager: chdir into MUSETALK_REPO_DIR for the duration of the with-block, then
    restore the original cwd -- see _musetalk_cwd_lock's docstring above for why this is scoped
    as tightly as possible rather than left changed permanently."""

    def __enter__(self) -> None:
        _musetalk_cwd_lock.acquire()
        self._previous_cwd = Path.cwd()
        os.chdir(MUSETALK_REPO_DIR)
        if str(MUSETALK_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(MUSETALK_REPO_DIR))

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            os.chdir(self._previous_cwd)
        finally:
            _musetalk_cwd_lock.release()


def _load_musetalk_models_sync() -> dict:
    global _musetalk_models
    if _musetalk_models is not None:
        return _musetalk_models

    with _MuseTalkCwdGuard():
        import torch
        from transformers import WhisperModel

        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.utils import load_all_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae, unet, pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="./models/musetalkV15/musetalk.json",
            device=device,
        )
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)

        # weight_dtype MUST be read after the .half() casts above (matches
        # scripts/realtime_inference.py's own ordering exactly) -- reading it right after
        # load_all_model() instead would capture unet's original float32 dtype, casting whisper
        # to float32 while the UNet itself is float16: a real dtype-mismatch crash
        # ("expected mat1 and mat2 to have the same dtype") found by actually running this
        # in-process, not just at import time.
        weight_dtype = unet.model.dtype

        audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        whisper = WhisperModel.from_pretrained("./models/whisper")
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)

        fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        timesteps = torch.tensor([0], device=device)

        _musetalk_models = {
            "vae": vae, "unet": unet, "pe": pe, "fp": fp, "device": device,
            "weight_dtype": weight_dtype, "audio_processor": audio_processor,
            "whisper": whisper, "timesteps": timesteps,
        }
        logger.info("[avatar_renderer] MuseTalk models loaded (device=%s)", device)
    return _musetalk_models


def _avatar_cache_paths(avatar_id: str) -> dict:
    avatar_dir = MUSETALK_RESULT_DIR / "avatars" / avatar_id
    return {
        "dir": avatar_dir,
        "full_imgs": avatar_dir / "full_imgs",
        "cache": avatar_dir / "cache.pkl",
    }


def _prepare_avatar_sync(avatar_id: str, motion_video_path: Path) -> dict:
    """Port of MuseTalk's own scripts/realtime_inference.py Avatar.prepare_material, adapted to
    use absolute paths/instance return values instead of argparse globals. Idempotent: returns
    the on-disk cache if avatar_id was already prepared (by this process or an earlier one --
    cache.pkl survives a process restart, matching the "durable, not in-memory" idempotency
    pattern used elsewhere in this pipeline, e.g. turn_publisher.py)."""
    paths = _avatar_cache_paths(avatar_id)
    if paths["cache"].exists():
        with open(paths["cache"], "rb") as f:
            return pickle.load(f)

    models = _load_musetalk_models_sync()
    with _MuseTalkCwdGuard():
        import cv2
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.preprocessing import get_landmark_and_bbox

        paths["full_imgs"].mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(motion_video_path))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(paths["full_imgs"] / f"{count:08d}.png"), frame)
            count += 1
        cap.release()

        input_img_list = sorted(glob.glob(str(paths["full_imgs"] / "*.png")))
        if not input_img_list:
            raise RuntimeError(f"No frames extracted from motion video {motion_video_path}")

        coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)

        extra_margin = 10
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        input_latent_list = []
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + extra_margin, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            input_latent_list.append(models["vae"].get_latents_for_unet(resized_crop_frame))

        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        mask_coords_list_cycle = []
        mask_list_cycle = []
        for frame, coord in zip(frame_list_cycle, coord_list_cycle):
            x1, y1, x2, y2 = coord
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=models["fp"], mode="jaw")
            mask_coords_list_cycle.append(crop_box)
            mask_list_cycle.append(mask)

        cache = {
            "coord_list_cycle": coord_list_cycle,
            "frame_list_cycle": frame_list_cycle,
            "input_latent_list_cycle": input_latent_list_cycle,
            "mask_coords_list_cycle": mask_coords_list_cycle,
            "mask_list_cycle": mask_list_cycle,
        }
        with open(paths["cache"], "wb") as f:
            pickle.dump(cache, f)
        logger.info("[avatar_renderer] MuseTalk avatar prepared and cached: avatar_id=%s frames=%d", avatar_id, len(frame_list))
    return cache


def _infer_sync(avatar_id: str, cache: dict, audio_path: Path, out_video_path: Path) -> None:
    """Port of MuseTalk's Avatar.inference -- the fast, per-utterance path that reuses the cached
    landmarks/latents/masks from _prepare_avatar_sync instead of recomputing them."""
    with _MuseTalkCwdGuard():
        import cv2
        import numpy as np
        import torch
        from musetalk.utils.blending import get_image_blending
        from musetalk.utils.utils import datagen

        models = _load_musetalk_models_sync()
        device, weight_dtype = models["device"], models["weight_dtype"]

        whisper_input_features, librosa_length = models["audio_processor"].get_audio_feature(
            str(audio_path), weight_dtype=weight_dtype
        )
        whisper_chunks = models["audio_processor"].get_whisper_chunk(
            whisper_input_features, device, weight_dtype, models["whisper"], librosa_length,
            fps=MUSETALK_FPS, audio_padding_length_left=2, audio_padding_length_right=2,
        )

        gen = datagen(whisper_chunks, cache["input_latent_list_cycle"], batch_size=8, device=device)
        res_frames = []
        with torch.no_grad():
            for whisper_batch, latent_batch in gen:
                audio_feature_batch = models["pe"](whisper_batch.to(device))
                latent_batch = latent_batch.to(device=device, dtype=models["unet"].model.dtype)
                pred_latents = models["unet"].model(
                    latent_batch, models["timesteps"], encoder_hidden_states=audio_feature_batch
                ).sample
                pred_latents = pred_latents.to(device=device, dtype=models["vae"].vae.dtype)
                recon = models["vae"].decode_latents(pred_latents)
                res_frames.extend(recon)

        if not res_frames:
            raise RuntimeError(f"MuseTalk produced no frames for {audio_path} (avatar_id={avatar_id})")

        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = out_video_path.parent / f"{out_video_path.stem}_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            coord_cycle = cache["coord_list_cycle"]
            frame_cycle = cache["frame_list_cycle"]
            mask_cycle = cache["mask_list_cycle"]
            mask_coord_cycle = cache["mask_coords_list_cycle"]

            for idx, res_frame in enumerate(res_frames):
                bbox = coord_cycle[idx % len(coord_cycle)]
                ori_frame = copy.deepcopy(frame_cycle[idx % len(frame_cycle)])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    continue
                mask = mask_cycle[idx % len(mask_cycle)]
                mask_crop_box = mask_coord_cycle[idx % len(mask_coord_cycle)]
                combined = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
                cv2.imwrite(str(tmp_dir / f"{idx:08d}.png"), combined)

            silent_video = out_video_path.parent / f"{out_video_path.stem}_silent.mp4"
            _run_ffmpeg([
                "ffmpeg", "-y", "-v", "warning", "-r", str(MUSETALK_FPS), "-f", "image2",
                "-i", str(tmp_dir / "%08d.png"), "-vcodec", "libx264", "-vf", "format=yuv420p",
                "-crf", "18", str(silent_video),
            ])
            _run_ffmpeg([
                "ffmpeg", "-y", "-v", "warning", "-i", str(audio_path), "-i", str(silent_video),
                "-c:v", "copy", "-c:a", "aac", str(out_video_path),
            ])
            silent_video.unlink(missing_ok=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_ffmpeg(args: List[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        logger.error("[avatar_renderer:ffmpeg] stderr (tail):\n%s", result.stderr[-2000:])
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode})")


# --------------------------------------------------------------------------------------
# Public async API -- called by realtime/avatar_speech.py
# --------------------------------------------------------------------------------------

async def ensure_avatar_ready(avatar_id: str, photo_path: Optional[Path] = None) -> None:
    """Idempotent setup for one exam attempt's avatar identity: renders the LivePortrait motion
    video once, then runs MuseTalk's one-time landmark/face-parsing/latent prep on it. Safe to
    call before every utterance -- both stages check their own on-disk cache first."""
    photo = photo_path or DEFAULT_REFERENCE_PHOTO
    motion_video = await asyncio.to_thread(_run_liveportrait_sync, avatar_id, photo)
    await asyncio.to_thread(_prepare_avatar_sync, avatar_id, motion_video)


async def render_utterance(avatar_id: str, audio_path: Path, out_video_path: Path) -> Path:
    """Renders one utterance's lip-synced video (with the source audio muxed in). Raises if
    ensure_avatar_ready hasn't completed for this avatar_id yet."""
    paths = _avatar_cache_paths(avatar_id)
    if not paths["cache"].exists():
        raise RuntimeError(f"avatar_id={avatar_id} is not prepared -- call ensure_avatar_ready first")

    with open(paths["cache"], "rb") as f:
        cache = pickle.load(f)

    start = time.monotonic()
    await asyncio.to_thread(_infer_sync, avatar_id, cache, audio_path, out_video_path)
    logger.info(
        "[avatar_renderer] MuseTalk utterance rendered in %.1fs avatar_id=%s -> %s",
        time.monotonic() - start, avatar_id, out_video_path,
    )
    return out_video_path
