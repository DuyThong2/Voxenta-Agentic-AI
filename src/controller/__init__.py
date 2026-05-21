from fastapi import APIRouter

from controller.auth import router as auth_router
from controller.home import router as home_router
from controller.webrtc import router as webrtc_router


router = APIRouter()
router.include_router(auth_router)
router.include_router(home_router)
router.include_router(webrtc_router)

