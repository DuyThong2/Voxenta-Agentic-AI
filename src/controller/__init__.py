from fastapi import APIRouter

from controller.auth import router as auth_router
from controller.home import router as home_router
from controller.webrtc import router as webrtc_router
from controller.archive_controller import router as archive_router
from controller.realtime_controller import router as realtime_router
from controller.avatar_webrtc_controller import router as avatar_webrtc_router


router = APIRouter()
router.include_router(auth_router)
router.include_router(home_router)
router.include_router(webrtc_router)
router.include_router(archive_router)
router.include_router(realtime_router)
router.include_router(avatar_webrtc_router)
