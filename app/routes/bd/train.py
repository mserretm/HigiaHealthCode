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
async def train_case(db: Session = Depends(get_db)):
    """
    Entrena el model amb casos clínics utilitzant dades de la base de dades.
    """
    try:
        # Obtenir casos pendents d'entrenament (T)
        pending_records = db.query(Case).filter(
            Case.us_registre == 'T',
            Case.us_estatentrenament == 0
        ).order_by(Case.cas).limit(10).all()
        
        if not pending_records:
            return {
                "status": "info",
                "message": "No hi ha casos pendents d'entrenament"
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
                    'cursclinic': record.cursclinic,
                    'dx_revisat': record.dx_revisat  # Mantener el valor original
                }
                
                # Verificar que dx_revisat no està buit
                if not case_data['dx_revisat']:
                    logger.warning(f"El cas {record.cas} té dx_revisat buit. S'actualitzarà l'estat a error.")
                    record.us_estatentrenament = 2  # Marcar com a error
                    db.commit()
                    continue
                
                # Processar entrenament
                await processor.process_train(case_data, db)
                
                # Actualitzar estat a la base de dades
                record.us_estatentrenament = 1
                db.commit()
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processant el cas {record.cas}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "message": f"Entrenament completat per {processed} casos",
            "processed_cases": processed
        }
        
    except Exception as e:
        logger.error(f"Error en l'entrenament: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en l'entrenament: {str(e)}"
        ) 