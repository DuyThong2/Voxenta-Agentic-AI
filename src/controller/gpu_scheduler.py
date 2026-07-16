"""GPU serialization between avatar rendering and YOLO proctoring (Phase 4 of
docs/realtime-self-hosted-avatar-plan.md).

Both controller/webrtc.py's YOLO proctoring inference and realtime/avatar_renderer.py's
LivePortrait/MuseTalk inference share one GPU; neither was written assuming concurrent GPU work
from the other. A single process-wide semaphore serializes the actual CUDA kernel launches so
they take turns instead of contending -- the exact scheduling *policy* (round-robin vs.
avatar-always-wins vs. proctoring auto-throttling) is still an open product question (see Open
Question 4 in the plan doc); this just provides the mechanism. Set
GPU_SCHEDULER_MAX_CONCURRENT=2+ on a multi-GPU pod where real overlap is possible -- not the case
on this single-GPU dev machine.
"""

import asyncio
import os

_MAX_CONCURRENT = max(1, int(os.getenv("GPU_SCHEDULER_MAX_CONCURRENT", "1")))
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


def gpu_lock() -> asyncio.Semaphore:
    """Usage: `async with gpu_scheduler.gpu_lock(): ...` around any GPU-bound call (avatar
    rendering, YOLO inference)."""
    return _semaphore
