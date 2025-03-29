import logging
from app.ml.engine import ModelEngine
from app.schemas.requests import (
    PredictRequest,
    TrainRequest,
    ValidateRequest,
    EvaluateRequest,
    ClinicalCase
)
from app.db.database import get_db
from sqlalchemy.orm import Session
from app.models.case import Case
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class CaseProcessor:
    def __init__(self):
        self.engine = ModelEngine()

    async def process_predict(self, data: dict) -> dict:
        try:
            result = self.engine.predict_case(data)
            return {
                "cas": data["cas"],
                "prediccions": result["prediccions"],
                "status": "success",
                "message": "Predicció completada correctament"
            }
        except Exception as e:
            logger.error(f"Error en la predicció del cas {data['cas']}: {str(e)}")
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "error",
                "message": str(e)
            }

    async def process_train(self, data: dict) -> dict:
        try:
            # Verificar que el caso existe en la base de datos
            db = next(get_db())
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")

            # Realizar el entrenamiento
            await self.engine.train_incremental(data)
            
            # Solo si el entrenamiento fue exitoso, actualizar el estado en la base de datos
            case.estat = "entrenat"
            db.commit()
            
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "success",
                "message": "Entrenament completat correctament"
            }
        except Exception as e:
            logger.error(f"Error en l'entrenament del cas {data['cas']}: {str(e)}")
            # En caso de error, asegurarnos de que el estado no se actualice
            if 'case' in locals():
                case.estat = "error"
                db.commit()
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "error",
                "message": str(e)
            }

    async def process_validate(self, data: dict) -> dict:
        try:
            # Verificar que el caso existe en la base de datos
            db = next(get_db())
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")

            # Realizar la validación
            metrics = self.engine.validate_model(data)
            
            # Solo si la validación fue exitosa, actualizar el estado
            case.estat = "validat"
            db.commit()
            
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "success",
                "message": "Validació completada correctament",
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Error en la validació del cas {data['cas']}: {str(e)}")
            # En caso de error, asegurarnos de que el estado no se actualice
            if 'case' in locals():
                case.estat = "error"
                db.commit()
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "error",
                "message": str(e)
            }

    async def process_evaluate(self, data: dict) -> dict:
        try:
            # Verificar que el caso existe en la base de datos
            db = next(get_db())
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")

            # Realizar la evaluación
            metrics = self.engine.validate_model(data)
            
            # Solo si la evaluación fue exitosa, actualizar el estado
            case.estat = "avaluat"
            db.commit()
            
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "success",
                "message": "Avaluació completada correctament",
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Error en l'avaluació del cas {data['cas']}: {str(e)}")
            # En caso de error, asegurarnos de que el estado no se actualice
            if 'case' in locals():
                case.estat = "error"
                db.commit()
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "error",
                "message": str(e)
            } 