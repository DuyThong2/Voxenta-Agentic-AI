"""Chay tac vu nen ma KHONG bi bo thu gom rac xoa giua chung -- va khong ro bo nho.

Python ghi ro trong tai lieu asyncio: event loop chi giu THAM CHIEU YEU toi task. Goi
`asyncio.create_task(...)` roi vut ket qua di la task co the bi GC xoa khi dang chay -- khong
loi, khong log, khong dau vet. Nguy hiem nhat la cho day luot noi BAI THI len Kafka: task bi xoa
giua chung nghia la mot luot noi bien mat khoi ban cham, va khong ai biet.

CAN THAN KHI CHAY TREN KUBERNETES/EKS -- ba dieu duoi day sinh ra tu chinh moi truong do:

1. Giu tham chieu manh nghia la mot task TREO VINH VIEN (goi HTTP khong timeout, broker khong
   phan hoi) se nam mai trong set thay vi duoc GC don. Pod song hang ngay, moi lan spawn them
   mot task treo la set phinh dan -> OOMKilled. Vi vay MOI task deu bi boc timeout, va set co
   nguong canh bao.

2. Rollout / scale-down / node drain deu gui SIGTERM roi SIGKILL sau
   terminationGracePeriodSeconds. Task dang chay bi cat ngang. `drain()` cho chung ket thuc
   trong mot han ngan -- goi tu lifespan cua FastAPI.

3. Han drain phai NGAN HON terminationGracePeriodSeconds (mac dinh k8s la 30s), khong thi
   SIGKILL van cat giua chung va drain chi lam cham viec tat pod.
"""

import asyncio
import logging
from typing import Any, Coroutine, Optional, Set

logger = logging.getLogger(__name__)

_background_tasks: Set[asyncio.Task] = set()

DEFAULT_TASK_TIMEOUT_SECONDS = 30.0
"""Tran cho MOT tac vu nen. Vuot thi huy -- tha mat mot lan day tin con hon ro bo nho.

30s la rong rai cho moi viec dang chay o day (publish Kafka, upload audio, goi HTTP noi bo).
Viec nao that su can lau hon thi truyen timeout rieng, dung bo han.
"""

DRAIN_TIMEOUT_SECONDS = 10.0
"""Cho tac vu nen ket thuc khi tat ung dung. PHAI ngan hon terminationGracePeriodSeconds."""

_WARN_TASK_COUNT = 200
"""Vuot nguong nay la co dau hieu task khong ket thuc kip -- canh bao truoc khi het bo nho."""


def spawn(
    coro: Coroutine[Any, Any, Any],
    *,
    name: Optional[str] = None,
    timeout: Optional[float] = DEFAULT_TASK_TIMEOUT_SECONDS,
) -> asyncio.Task:
    """Thay cho `asyncio.create_task` o moi cho chay-roi-quen."""
    task = asyncio.create_task(_guarded(coro, timeout), name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_done)
    if len(_background_tasks) >= _WARN_TASK_COUNT:
        logger.warning(
            "[background] dang co %d tac vu nen chua ket thuc -- kiem tra xem co gi bi treo",
            len(_background_tasks),
        )
    return task


async def _guarded(coro: Coroutine[Any, Any, Any], timeout: Optional[float]) -> Any:
    if timeout is None:
        return await coro
    return await asyncio.wait_for(coro, timeout)


def _on_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exception = task.exception()
    if isinstance(exception, asyncio.TimeoutError):
        logger.error("[background] tac vu nen %s qua han va bi huy", task.get_name())
    elif exception is not None:
        logger.error(
            "[background] tac vu nen %s that bai", task.get_name(), exc_info=exception
        )


async def drain(timeout: float = DRAIN_TIMEOUT_SECONDS) -> None:
    """Cho tac vu nen ket thuc khi tat ung dung. Goi tu lifespan cua FastAPI."""
    pending = set(_background_tasks)
    if not pending:
        return
    logger.info("[background] cho %d tac vu nen ket thuc truoc khi tat", len(pending))
    done, still_pending = await asyncio.wait(pending, timeout=timeout)
    if still_pending:
        # Khong the cho lau hon: SIGKILL cua k8s se toi truoc. Huy va ghi lai de con truy.
        logger.warning(
            "[background] %d tac vu nen chua xong sau %.0fs -- huy", len(still_pending), timeout
        )
        for task in still_pending:
            task.cancel()
