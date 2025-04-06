import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.schemas.requests import TrainRequest
from app.services.case_processor import CaseProcessor
from app.db.session import get_db
from app.models.case import Case
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)
processor = CaseProcessor()

@router.post("/")
async def train_case(data: TrainRequest, db: Session = Depends(get_db)):
    """
    Entrena el model amb un cas clínic utilitzant dades en format JSON.
    """
    try:
        case_data = data.dict()
        
        # Realitzar entrenament
        result = await processor.process_train(case_data, db)
        
        # Guardar registre a la base de dades
        db_case = Case(
            cas=case_data['cas'],
            dx_revisat=case_data['dx_revisat'],
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
            us_registre='T',  # Per entrenament
            us_estatentrenament=1,  # Processat
            us_dataentrenament=datetime.now()
        )
        
        db.add(db_case)
        db.commit()
        
        return result
    except Exception as e:
        logger.error(f"Error en l'entrenament del cas {data.cas}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en l'entrenament: {str(e)}"
        ) 