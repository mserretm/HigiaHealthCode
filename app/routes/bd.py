from fastapi import APIRouter, HTTPException
from app.schemas.case import Case, CaseResponse
from app.services.case_processor import CaseProcessor
from typing import List

router = APIRouter()
case_processor = CaseProcessor()

@router.post("/predict", response_model=CaseResponse)
async def predict_case(case: Case):
    try:
        result = await case_processor.process_predict(case)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=CaseResponse)
async def train_case(case: Case):
    try:
        result = await case_processor.process_train(case)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate", response_model=CaseResponse)
async def validate_case(case: Case):
    try:
        result = await case_processor.process_validate(case)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
