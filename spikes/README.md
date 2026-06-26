# Phase 0 spikes — setup notes

Throwaway PoC scripts for `docs/realtime-self-hosted-avatar-plan.md`'s Phase 0. Nothing here is
wired into the FastAPI app (`src/app.py`) — these are standalone scripts you run by hand to prove
feasibility before any of this gets productionized in Phase 3/4. Fill in the credentials below
when you're ready; until then the scripts run and clearly tell you what's missing (they don't
fail with an unrelated stack trace).

## 1. Azure (for `voice_live_poc.py` and, later, TTS in Phase 4) — DONE

This repo already uses Azure Speech for **archival** STT (`AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION`
in the root `.env`, consumed by `src/utils/speech_client.py`) — that part needs no new setup.

**Azure Voice Live (realtime STT+VAD) is a separate surface, accessed via the
`azure-ai-voicelive` SDK** (now a real `pyproject.toml` dependency, added via
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

## 2. Hugging Face (for `avatar_render_poc.py`)

Both LivePortrait and MuseTalk ship their weights on the Hugging Face Hub, not PyPI.

1. Create a free account at huggingface.co if you don't have one.
2. (Only needed if a model page says "gated"/requires accepting terms — neither of the two repos
   below currently does, but re-check the model pages.) Generate an access token under
   *Settings → Access Tokens*, then `hf auth login` once, or set `HF_TOKEN` in `.env`.
3. CLI: use `uv` instead of `pip` — either ephemeral (no project pollution) via
   `uvx --from "huggingface_hub[cli]" hf download ...`, or `uv add --group dev "huggingface_hub[cli]"`
   then `uv run hf download ...` if you want it reusable. Note: the old `huggingface-cli` command
   is deprecated — current CLI entrypoint is `hf` (e.g. `hf download`, `hf auth login`).
4. Download weights (verified working repo names as of this writing):
   ```bash
   # LivePortrait (official KwaiVGI repo)
   hf download KwaiVGI/LivePortrait --local-dir agents/spikes/weights/liveportrait

   # MuseTalk (official TMElyralab repo)
   hf download TMElyralab/MuseTalk --local-dir agents/spikes/weights/musetalk
   ```
   MuseTalk's own repo additionally expects sibling model folders (dwpose, face-parse-bisent,
   sd-vae-ft-mse, whisper) in a specific directory layout — check `TMElyralab/MuseTalk`'s own
   README on Hugging Face/GitHub for the exact layout expected by its inference script before
   wiring `avatar_render_poc.py`'s real calls in.
5. If Hugging Face is unreachable from your network, you can mirror via
   `HF_ENDPOINT=https://hf-mirror.com` before running the same `hf download` command.

`avatar_render_poc.py` reads `LIVEPORTRAIT_WEIGHTS_DIR`/`MUSETALK_WEIGHTS_DIR` from `.env` (defaults
to the paths above) and, like the Voice Live script, stubs the actual LivePortrait/MuseTalk
inference calls with a `NotImplementedError` — the exact Python entrypoint depends on which
version of each repo's inference script you end up cloning alongside the weights, so it's left for
whoever wires this up once the weights are actually downloaded.

## 2b. GitHub repos — installed straight into `agents/.venv` (the main project venv)

Weights alone aren't enough — the actual inference code lives in each project's GitHub repo
(Hugging Face only hosts the weight *data*). Clone both (shallow, no need for full history):

```bash
cd agents
git clone --depth=1 https://github.com/KwaiVGI/LivePortrait spikes/repos/LivePortrait
git clone --depth=1 https://github.com/TMElyralab/MuseTalk spikes/repos/MuseTalk
```

**Current state (2026-06-25): there is only one venv.** Earlier in this project's history,
LivePortrait/MuseTalk's pinned dependencies looked mutually incompatible with the main app's
(`torch==2.3.0` vs `2.5.1`, `numpy<2` vs `>=1.26.0`→`2.3.3`, `opencv-python` GUI build vs
`opencv-python-headless`, plus `tensorflow==2.12.0`), so a separate `spikes/.venv-avatar` was
built by hand to isolate the risk. That separate venv proved the pipeline works end-to-end on
this GPU — but per a deliberate later decision, **Phase 4's `avatar_renderer.py` will run inside
the main app's own process/venv, not a separate one**, so all of this is now declared directly
in `agents/pyproject.toml` instead, and `agents/.venv` is the only venv this PoC needs. The table
below is kept for historical context (why each version was originally pinned where it was) —
none of it requires manual action anymore, every fix is already in `agents/pyproject.toml`:

| package | original conflict | how it's resolved now |
|---|---|---|
| python | MuseTalk's `tensorflow==2.12.0` has no cp312 wheel | whole project pinned to 3.11 (`.python-version`, `requires-python`) |
| torch | LivePortrait pins `2.3.0`, main app was at `2.5.1` | main app's torch **downgraded to `2.3.0`** — mmcv's compiled extension is built against torch2.3.0 specifically and failed to load (`ImportError: DLL load failed while importing _ext`) under `2.5.1`; confirmed by actually running the pipeline, not just an import check. `ultralytics` only needs `torch>=1.8.0`, so this wasn't a real constraint conflict. |
| numpy | LivePortrait needs `<2`, tensorflow declares `<1.24`, main app floated to `2.3.3` | pinned `numpy==1.26.4` + `[tool.uv] override-dependencies` to force it past tensorflow's looser bound |
| opencv | `opencv-python` (GUI) vs `opencv-python-headless` | only `-headless` installed; the GUI variant silently overwrites shared files if both land in the same venv, so never install both |
| mmcv vs mmdet/mmpose | mmdet/mmpose hard-assert mmcv `<2.2.0`, only available prebuilt wheel is `2.2.0` | vendored, patched mmdet/mmpose wheels in `agents/vendor/` (see `pyproject.toml`'s `[tool.uv.sources]`) — pure-Python, no manual site-packages editing needed anymore |
| tensorflow on Windows | `tensorflow` itself is an empty stub on Windows | `tensorflow-intel` declared explicitly (uv doesn't pull it in automatically) |
| jax (via tensorflow-intel) | unpinned resolved to a numpy<2.0-incompatible version | pinned `jax==0.4.13` via override-dependencies |
| transformers | unpinned resolved to a breaking major-version jump (5.x) | pinned `transformers==4.39.2` (MuseTalk's tested version) |
| setuptools | newer releases removed `pkg_resources`, which mmengine needs | upper-bounded `<80` |
| onnxruntime-gpu CUDA provider | its prebuilt wheel needs CUDA **11.x** DLLs, but torch's `cu121` build only bundles CUDA 12.x | torch/torchvision/torchaudio + mmcv all moved to the **cu118** index instead (matches what onnxruntime-gpu==1.18.0 actually needs) |

So setup is now just:
```bash
cd agents
uv sync
```

**Weight directory junctions** (one-time; lets each repo find the already-downloaded weights at
their expected relative paths, without duplicating multi-GB files):
```powershell
New-Item -ItemType Junction -Path "spikes\repos\LivePortrait\pretrained_weights" -Target "spikes\weights\liveportrait"
New-Item -ItemType Junction -Path "spikes\repos\MuseTalk\models" -Target "spikes\weights\musetalk"
```
(Confirmed via reading the actual source: LivePortrait's `src/config/inference_config.py` and
`crop_config.py` expect `<repo>/pretrained_weights/liveportrait/...` and
`<repo>/pretrained_weights/insightface/...`; MuseTalk's `musetalk/utils/face_parsing/__init__.py`
and `scripts/inference.py`'s CLI defaults expect `<repo>/models/...` — both match the structure
`hf download` already produced under `spikes/weights/`.)

Then just run the PoC like any other script in this venv:

```bash
uv run python spikes/avatar_render_poc.py --audio data/sample.wav --photo data/image.png
```

(No more separate-venv invocation — `voice_live_poc.py` and `avatar_render_poc.py` both run the
same way now, via `uv run` against `agents/.venv`.)

**Timing history (all real runs, same 3.22s audio clip, output video confirmed valid each time —
`spikes/outputs/musetalk/v15/image--d0_sample.mp4`, 80 frames @ 25fps = 3.2s, 740x492):**
1. First successful run (separate `.venv-avatar`, cold disk cache): LivePortrait 19.9s + MuseTalk
   196.5s = **216.4s end-to-end** (~67x slower than realtime).
2. After unifying numpy/opencv between the two venvs (still separate venvs, warm disk cache):
   LivePortrait 21.75s + MuseTalk 57.8s = **79.5s end-to-end** — most of the improvement is
   likely warm OS disk cache for the multi-GB checkpoints, not the dependency unification itself.
3. After the cu118 switch, still via the now-retired separate venv, disk cache cold again:
   LivePortrait 28.5s + MuseTalk 165.5s = **194.1s end-to-end**. (onnxruntime was assumed fixed
   at this point — turned out not to be, see the GPU section above; this number does NOT reflect
   a working onnxruntime CUDA provider.)
4. Through `agents/.venv` directly (the only venv now), torch==2.3.0+cu118, run twice back to
   back with no config change: **177s cold disk cache** (LivePortrait 37.1s + MuseTalk 139.9s)
   then **82s warm disk cache** (22.5s + 59.5s) for the identical setup. This nearly 2x swing
   from disk-cache state alone (not a CUDA-variant difference — same exact config both times) is
   the clearest evidence yet of how much cold-start dominates every number in this list.
None of these are warm-*process* numbers (every run is a fresh subprocess reloading all
checkpoints from disk) — that's still an open item. MuseTalk's PyTorch-based UNet inference
dominates the total regardless of any of the onnxruntime/CUDA-variant work above — even the best
number so far (82s for a 3.22s clip, ~25x slower than realtime) is nowhere close to realtime, and
nothing in this dependency-consolidation work was expected to (or did) change that verdict.

**Status (2026-06-25): fully resolved, declarative, single venv — `spikes/` has no
`pyproject.toml`/`uv.lock`/`.venv` of its own anymore, by design.** Everything (vendored
mmdet/mmpose wheels, numpy/jax overrides, transformers/setuptools pins, the cu118 switch, the
torch==2.3.0 pin) lives only in `agents/pyproject.toml` — there is exactly one venv
(`agents/.venv`) for the whole project, including these PoC scripts. The earlier separate
`spikes/.venv-avatar`/`spikes/pyproject.toml` setup (described above, kept for historical
context) has been fully retired and deleted. Running the actual pipeline through `agents/.venv`
surfaced one more real issue beyond the import smoke test: mmcv's compiled extension failed to
load under the venv's then-current `torch==2.5.1` (`ImportError: DLL load failed while importing
_ext`) — fixed by downgrading `agents/pyproject.toml`'s torch/torchvision/torchaudio to exactly
`2.3.0`/`0.18.0`/`2.3.0` (matching what mmcv/LivePortrait actually need; OpenMMLab doesn't
publish mmcv wheels for any newer torch version on any CUDA index). There is no more
hand-patching of installed packages needed anywhere. Treat `agents/pyproject.toml` as the single
source of truth for these dependencies going forward (it has the same fixes, with inline
comments explaining each one — read those before changing anything dependency-related here).

(Historical note for whoever writes the eventual EKS Dockerfile: the vendored-wheel fix for
mmdet/mmpose in `agents/vendor/` is pure-Python and platform-independent, so it should carry over
to Linux unchanged. The cu118 CUDA choice was driven by `onnxruntime-gpu==1.18.0`'s prebuilt
wheel needing CUDA-11-flavored DLLs specifically — re-verify that's still true if this version
ever changes.)

## 3. GPU — real GPU confirmed, torch pinned to CUDA 11.8 (not 12.1) — DONE

`nvidia-smi` confirms a real NVIDIA GPU on this machine: **RTX 2000 Ada Generation, ~8GB VRAM,
driver 595.71, CUDA 13.2 supported**. Open Question 2 in the plan doc (where to run Phase 0) is
resolved: locally, on this GPU.

**Current state, fixed durably in `agents/pyproject.toml`**: explicit `torch==2.3.0`/
`torchvision==0.18.0`/`torchaudio==2.3.0` entries in `[project.dependencies]` (downgraded from
`2.5.1`/`0.20.1`/`2.5.1` — see the torch row in the table above for why), plus
`[tool.uv.sources]` pinning all three (and `mmcv`) to a `[[tool.uv.index]]` named
`pytorch-cu118`/`openmmlab-cu118-torch230`. Recreate/sync the venv with:
```powershell
cd D:\semester9\agents
Remove-Item -Recurse -Force .venv   # only needed if the venv already exists in a stale state
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```
Must print a version ending in `+cu118` (not `+cpu`, and not `+cu121` either — see below), `True`,
and `NVIDIA RTX 2000 Ada Generation Laptop GPU`. Also re-verify `ultralytics`/YOLO (the main
app's proctoring dependency) still imports and loads a model fine after the torch downgrade.

**Why CUDA 11.8 and not 12.1 (changed 2026-06-25, was cu121 originally):** `onnxruntime-gpu==1.18.0`
(needed for LivePortrait's face-detection/landmark step) only ships a stable PyPI build for
CUDA 11.x — confirmed directly via `pefile` on `onnxruntime_providers_cuda.dll`'s import table
(`cublas64_11.dll`/`cudart64_110.dll`/`cufft64_10.dll`). A `cu121` torch build only bundles
CUDA-12 DLLs, so onnxruntime's CUDA execution provider could never load
(`LoadLibrary failed with error 126`) and silently fell back to CPU no matter what
`PATH`/`os.add_dll_directory()` said — that's a missing *dependency-of-a-dependency*, not a
PATH problem. onnxruntime-gpu only publishes CUDA-12 builds via an unstable nightly-only Azure
DevOps feed (no stable release), so matching torch to CUDA 11.8 instead — which the GPU driver
runs just fine, newer drivers are always backwards-compatible with older CUDA runtimes — is the
durable fix. `mmcv`'s index moved to the matching `cu118`/`torch2.3.0` build too (it has its own
compiled CUDA extension and needs to match what torch actually bundles).

**Pitfall hit while making this switch, worth knowing about for any future index change:** after
flipping `mmcv`'s `[tool.uv.sources]` index from the cu121 build to the cu118 one, `import mmcv`
kept crashing (`ImportError: DLL load failed while importing _ext`) even with torch correctly on
cu118 — `pefile` on the installed `mmcv/_ext.*.pyd` showed it still importing `cudart64_12.dll`,
i.e. the *old* cu121-built file, not the cu118 one this index actually serves. A plain `uv sync`
does not always notice that a `[tool.uv.sources]` index changed for an already-resolved exact
version on a `flat`-format index (no per-wheel hash to compare against) — it kept silently
serving the stale cu121 wheel from cache. Fix: `uv sync --reinstall-package mmcv
--link-mode=copy` forced a genuine re-fetch from the new index; afterward `_ext.*.pyd` correctly
imports `cudart64_110.dll` and `mmcv`/`mmdet`/`mmpose` import and work correctly. If you ever
change which index a pinned-version package should resolve from, force
`--reinstall-package <name>` once rather than trusting a plain `uv sync` to notice.

**However, onnxruntime's CUDA provider still doesn't actually work, for a different and deeper
reason** (re-confirmed directly in `agents/.venv` after the mmcv fix above): loading
`onnxruntime_providers_cuda.dll` directly (bypassing onnxruntime's own wrapper, via
`ctypes.WinDLL`, with `torch/lib` already registered via `os.add_dll_directory`) fails with
`WinError 1114: A dynamic link library (DLL) initialization routine failed` — a *different*
error than the original "error 126 / missing dependency." All 5 of its direct DLL dependencies
(`cublasLt64_11.dll`/`cublas64_11.dll`/`cudnn64_8.dll`/`cufft64_10.dll`/`cudart64_110.dll`) load
individually without error, so this is the provider DLL's own init code crashing — most likely
because torch's bundled `cudnn64_8.dll` (cuDNN 8.7.x) is a different exact patch version than
whatever onnxruntime-gpu==1.18.0 was actually built/tested against, despite sharing the same
filename. **Not pursued further: onnxruntime permanently falls back to CPU for the
face-detection step, and that's an accepted state** — it's a small fraction of LivePortrait's
already-much-smaller stage (37s of 177s total); MuseTalk (139.9s, unaffected either way) is what
actually dominates. This does **not** change the realtime verdict below.

**Why this needed to become a `pyproject.toml` change and not just a one-off
`uv pip install --reinstall`:** a bare version pin like `torch==2.3.0` doesn't disambiguate CPU
vs. CUDA build tags — `uv`/`pip` may consider an already-installed CPU wheel as "satisfying" that
constraint. Worse, `uv run`/`uv sync` always re-resolve against `pyproject.toml`/`uv.lock` before
running anything — so a manual `--reinstall` fix that isn't reflected in `pyproject.toml` gets
**silently reverted on the very next `uv run`** (this happened once already in this project). The
`[tool.uv.sources]`/`[tool.uv.index]` pins above are what make the intended CUDA build the actual
locked resolution, so they survive every future `uv sync`/`uv run`.

8GB VRAM is modest for running LivePortrait+MuseTalk together — if that turns out to be
insufficient, quantization (fp16 for `.pth` checkpoints, int8 via onnxruntime for `.onnx`
models) is an acceptable fallback to try before concluding the approach doesn't fit, per the
plan doc's Phase 0 go/no-go gate notes.

**Platform tradeoff worth knowing about:** pinning torch/torchvision/torchaudio to the
`pytorch-cu118` index means `uv sync` always fetches the (larger) CUDA wheel, even on a machine
with no GPU — it still installs and imports fine there (`torch.cuda.is_available()` just returns
`False`, no crash), just a bigger download for no benefit. The real risk is platform-availability,
not correctness: if this project is ever set up on a platform PyTorch doesn't publish `cu118`
wheels for (macOS, ARM), `uv sync` would fail to find a matching wheel at all. Not a concern for
this Windows+NVIDIA dev machine today, but worth remembering if that ever changes.

## 4. Running

**Use `uv run`, not a bare `python`/`pip install`** — `azure-ai-voicelive` lives in
`agents/.venv` (added via `uv add`), not in any global Python install. If you run
`python spikes/voice_live_poc.py` directly with a global interpreter, the env-var check will
look fine but it'll then fail with `ModuleNotFoundError: azure.ai.voicelive`.

```bash
cd agents

# huggingface_hub[cli]/psutil — fine to install however you like (uvx, global pip, etc.),
# they're not imported by anything that needs to run inside agents/.venv.
uv add --group dev "huggingface_hub[cli]" psutil   # or: pip install -r spikes/requirements-poc.txt

# Already-correct-format test WAVs exist in data/ (16kHz/16-bit/mono, confirmed via `wave`):
#   data/sample.wav       (3.22s)
#   data/sampleError.wav  (4.74s)
#   data/sampleError2.wav (6.04s)
# data/note.txt has the ffmpeg command used to make sample.wav from a .m4a recording, in case
# you want to add more.
uv run python spikes/voice_live_poc.py --wav data/sample.wav

# No reference photo exists in data/ yet for avatar_render_poc.py — supply your own (a clear
# front-facing photo). The driving audio can reuse the same sample.
uv run python spikes/avatar_render_poc.py --audio data/sample.wav --photo data/image.png
```

Results/benchmarks get appended to `spikes/poc-results.md`.
