import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.schemas.requests import EvaluateRequest
from app.services.case_processor import CaseProcessor
from app.db.session import get_db
from app.models.case import Case
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)
processor = CaseProcessor()

@router.post("/")
async def evaluate_case(data: EvaluateRequest, db: Session = Depends(get_db)):
    """
    Avalua el model amb un cas clínic utilitzant dades en format JSON.
    """
    try:
        case_data = data.dict()
        
        # Realitzar avaluació
        result = await processor.process_evaluate(case_data)
        
        # Calcular precisió si tenim el diagnòstic real
        accuracy = 0.0
        if 'dx_revisat' in case_data and case_data['dx_revisat']:
            predicted_dx = result['prediccions'][0]['diagnostic'] if result['prediccions'] else None
            actual_dx = case_data['dx_revisat'][0] if case_data['dx_revisat'] else None
            accuracy = 1.0 if predicted_dx == actual_dx else 0.0
        
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
            dx_revisat=case_data.get('dx_revisat', ''),
            us_registre='E',  # Per avaluació
            us_estatentrenament=1,  # Processat
            dx_prediccio=result['prediccions'][0]['diagnostic'] if result['prediccions'] else None,
            us_dataentrenament=datetime.now()
        )
        
        db.add(db_case)
        db.commit()
        
        # Afegir mètriques a la resposta
        response = result.copy()
        response['metrics'] = {
            'accuracy': accuracy,
            'predicted_diagnostic': result['prediccions'][0]['diagnostic'] if result['prediccions'] else None,
            'actual_diagnostic': case_data.get('dx_revisat', [None])[0] if case_data.get('dx_revisat') else None
        }
        
        return response
    except Exception as e:
        logger.error(f"Error en l'avaluació del cas {data.cas}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en l'avaluació: {str(e)}"
        ) 