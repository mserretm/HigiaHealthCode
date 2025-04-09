from fastapi import APIRouter
from .train import router as train_router
from .validate import router as validate_router

router = APIRouter()

router.include_router(train_router, prefix="/train", tags=["train"])
router.include_router(validate_router, prefix="/validate", tags=["validate"])