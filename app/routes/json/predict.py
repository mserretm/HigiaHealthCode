import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.schemas.requests import PredictRequest
from app.services.case_processor import CaseProcessor
from app.db.session import get_db
from app.models.case import Case
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)
processor = CaseProcessor()

@router.post("/")
async def predict_case(data: PredictRequest, db: Session = Depends(get_db)):
    """
    Realitza prediccions per un cas clínic utilitzant dades en format JSON.
    """
    try:
        case_data = data.dict()
        
        # Realitzar predicció
        result = await processor.process_predict(case_data)
        
        # Guardar registre a la base de dades
        db_case = Case(
            cas=case_data['cas'],
            edat=case_data['edat'],
            genere=case_data['genere'],
            c_alta=case_data['c_alta'],
            periode=case_data['periode'],
            servei=case_data['servei'],
            motiuingres=case_data['motiuingres'],
            malaltiaactual=case_data['malaltiaactual'],
            exploracio=case_data['exploracio'],
            provescomplementariesing=case_data['provescomplementariesing'],
            provescomplementaries=case_data['provescomplementaries'],
            evolucio=case_data['evolucio'],
            antecedents=case_data['antecedents'],
            cursclinic=case_data['cursclinic'],
            us_registre='E',  # Per avaluació
            us_estatentrenament=1,  # Processat
            dx_prediccio='|'.join([p['code'] for p in result['prediccions']]),
            us_dataentrenament=datetime.now()
        )
        
        db.add(db_case)
        db.commit()
        
        return result
    except Exception as e:
        logger.error(f"Error en la predicció del cas {data.cas}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicció: {str(e)}"
        ) 