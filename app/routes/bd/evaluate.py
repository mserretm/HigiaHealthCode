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
async def evaluate_case(db: Session = Depends(get_db)):
    """
    Avalua el model amb casos clínics utilitzant dades de la base de dades.
    """
    try:
        # Obtenir casos pendents d'avaluació
        pending_records = db.query(Case).filter(
            Case.us_registre == 'E',
            Case.us_estatentrenament == 1
        ).order_by(Case.cas).limit(100).all()
        
        if not pending_records:
            return {
                "status": "info",
                "message": "No hi ha casos pendents d'avaluació"
            }
        
        # Processar cada cas
        processed = 0
        results = []
        total_accuracy = 0
        
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
                    'dx_revisat': record.dx_revisat.split('|') if record.dx_revisat else []
                }
                
                # Processar avaluació
                result = await processor.process_evaluate(case_data)
                
                # Calcular precisió per aquest cas
                predicted_dx = result['prediccions'][0]['diagnostic'] if result['prediccions'] else None
                actual_dx = case_data['dx_revisat'][0] if case_data['dx_revisat'] else None
                accuracy = 1.0 if predicted_dx == actual_dx else 0.0
                total_accuracy += accuracy
                
                # Guardar resultats
                results.append({
                    "cas": case_data['cas'],
                    "prediccions": result['prediccions'],
                    "diagnostic_real": case_data['dx_revisat'],
                    "accuracy": accuracy
                })
                
                # Actualitzar estat a la base de dades
                record.dx_prediccio = predicted_dx
                db.commit()
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error processant el cas {record.cas}: {str(e)}")
                continue
        
        # Calcular precisió global
        global_accuracy = total_accuracy / processed if processed > 0 else 0
        
        return {
            "status": "success",
            "message": f"Avaluació completada per {processed} casos",
            "processed_cases": processed,
            "global_accuracy": global_accuracy,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error en l'avaluació: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en l'avaluació: {str(e)}"
        ) 