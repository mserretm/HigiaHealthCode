import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.case_processor import CaseProcessor
from app.models.case import Case
from app.schemas.case import CaseBase

router = APIRouter()
logger = logging.getLogger(__name__)
processor = CaseProcessor()

@router.post("/")
async def validate_case(db: Session = Depends(get_db)):
    """
    Valida el model amb casos clínics utilitzant dades de la base de dades.
    """
    try:
        # Obtenir casos pendents de validació (V)
        pending_records = db.query(Case).filter(
            Case.us_registre == 'V',
            Case.us_estatentrenament == 0
        ).order_by(Case.cas).limit(100).all()
        
        if not pending_records:
            return {
                "status": "info",
                "message": "No hi ha casos pendents de validació"
            }
        
        # Processar cada cas
        processed = 0
        total_metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0
        }
        
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
                    'dx_revisat': record.dx_revisat if record.dx_revisat else ''
                }
                
                # Processar validació
                result = await processor.process_validate(case_data, db)
                
                # Actualitzar estat a la base de dades
                record.us_estatentrenament = 1
                db.commit()
                
                # Acumular mètriques
                if 'metrics' in result and 'classification' in result['metrics']:
                    for metric in total_metrics:
                        if metric in result['metrics']['classification']:
                            total_metrics[metric] += result['metrics']['classification'][metric]
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processant el cas {record.cas}: {str(e)}")
                continue
        
        # Calcular mitjanes
        if processed > 0:
            for metric in total_metrics:
                total_metrics[metric] /= processed
        
        return {
            "status": "success",
            "message": f"Validació completada per {processed} casos",
            "processed_cases": processed,
            "metrics": total_metrics
        }
        
    except Exception as e:
        logger.error(f"Error en la validació: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en la validació: {str(e)}"
        )

@router.post("/single/")
async def validate_single_case(case: CaseBase, db: Session = Depends(get_db)):
    """
    Valida un cas clínic específic amb el model entrenat.
    """
    try:
        result = await processor.process_validate(case.dict(), db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 