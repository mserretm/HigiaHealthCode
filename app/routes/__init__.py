from fastapi import APIRouter
from .json import router as json_router
from .bd import router as bd_router

router = APIRouter()
router.include_router(json_router, prefix="/json")
router.include_router(bd_router, prefix="/bd")
