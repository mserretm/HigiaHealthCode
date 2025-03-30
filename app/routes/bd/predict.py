import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.case_processor import CaseProcessor
from app.models.case import Case

router = APIRouter()
logger = logging.getLogger(__name__)
processor = CaseProcessor()

@router.post("/")
async def predict_case(db: Session = Depends(get_db)):
    """
    Realitza prediccions per casos clínics utilitzant dades de la base de dades.
    """
    try:
        # Obtenir casos pendents de predicció (E)
        pending_records = db.query(Case).filter(
            Case.us_registre == 'E',
            Case.us_estatentrenament == 0
        ).order_by(Case.cas).limit(100).all()
        
        if not pending_records:
            return {
                "status": "info",
                "message": "No hi ha casos pendents de predicció"
            }
        
        # Processar cada cas
        processed = 0
        for record in pending_records:
            try:
                # Convertir registre a diccionari
                case_data = {
                    'cas': record.cas,
                    'edat': record.edat,
                    'genere': record.genere,
                    'c_alta': record.c_alta,
                    'periode': record.periode,
                    'servei': record.servei,
                    'motiuingres': record.motiuingres,
                    'malaltiaactual': record.malaltiaactual,
                    'exploracio': record.exploracio,
                    'provescomplementariesing': record.provescomplementariesing,
                    'provescomplementaries': record.provescomplementaries,
                    'evolucio': record.evolucio,
                    'antecedents': record.antecedents,
                    'cursclinic': record.cursclinic
                }
                
                # Processar predicció
                result = await processor.process_predict(case_data)
                
                # Guardar predicció a la base de dades
                record.us_estatentrenament = 1
                record.dx_prediccio = '|'.join([pred['code'] for pred in result['prediccions']])
                db.commit()
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processant el cas {record.cas}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "message": f"Prediccions completades per {processed} casos",
            "processed_cases": processed
        }
        
    except Exception as e:
        logger.error(f"Error en les prediccions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en les prediccions: {str(e)}"
        ) 