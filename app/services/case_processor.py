import logging
from app.ml.engine import ModelEngine
from app.schemas.requests import (
    PredictRequest,
    TrainRequest,
    ValidateRequest,
    ClinicalCase
)
from app.db.database import get_db
from sqlalchemy.orm import Session
from app.models.case import Case
from fastapi import HTTPException
from typing import List

logger = logging.getLogger(__name__)

class CaseProcessor:
    def __init__(self):
        self.engine = ModelEngine()

    def get_pending_cases(self, db: Session) -> List[Case]:
        """
        Obtén els casos pendents de predicció.
        """
        return db.query(Case).filter(
            Case.estat == "pendent",
            Case.dx_revisat.isnot(None)
        ).all()

    async def process_predict(self, data: dict) -> dict:
        try:
            # Verificar que el cas existeix en la base de dades
            db = next(get_db())
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")

            # Verificar que el cas està pendent
            if case.estat != "pendent":
                return {
                    "cas": data["cas"],
                    "prediccions": [],
                    "status": "info",
                    "message": f"El cas {data['cas']} ja ha estat processat (estat: {case.estat})"
                }

            # Realitzar la predicció
            result = self.engine.predict_case(data)
            
            # Actualitzar l'estat i les prediccions en la base de dades
            case.estat = "predit"
            case.dx_prediccio = "|".join([p["code"] for p in result["prediccions"]])
            db.commit()
            
            return {
                "cas": data["cas"],
                "prediccions": result["prediccions"],
                "status": "success",
                "message": "Predicció completada correctament"
            }
        except Exception as e:
            logger.error(f"Error en la predicció del cas {data['cas']}: {str(e)}")
            # En cas d'error, actualizar l'estat
            if 'case' in locals():
                case.estat = "error"
                db.commit()
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "error",
                "message": str(e)
            }

    async def get_pending_predictions(self) -> dict:
        """
        Obtén la llista de casos pendents de predicció.
        """
        try:
            db = next(get_db())
            pending_cases = self.get_pending_cases(db)
            
            if not pending_cases:
                return {
                    "status": "info",
                    "message": "No hi ha casos pendents de predicció",
                    "pending_cases": []
                }
            
            return {
                "status": "success",
                "message": f"S'han trobat {len(pending_cases)} casos pendents",
                "pending_cases": [case.cas for case in pending_cases]
            }
        except Exception as e:
            logger.error(f"Error obtenint casos pendents: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "pending_cases": []
            }

    async def process_train(self, data: dict) -> dict:
        try:
            # Verificar que el cas existeix en la base de dades
            db = next(get_db())
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")

            # Realitzar l'entrenament
            await self.engine.train_incremental(data)
            
            # Només si l'entrenament fou exitós, actualizar l'estat en la base de dades
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
            # En cas d'error, assegurar-nos de que l'estat no es actualitzi
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
            # Verificar que el cas existeix en la base de dades
            db = next(get_db())
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")

            # Realitzar la validació
            metrics = self.engine.validate_model(data)
            
            # Només si la validació fou exitosa, actualizar l'estat
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
            # En cas d'error, assegurar-nos de que l'estat no es actualitzi
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
            # Verificar que el cas existeix en la base de dades
            db = next(get_db())
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")

            # Realitzar la avaluació
            metrics = self.engine.validate_model(data)
            
            # Només si la avaluació fou exitosa, actualizar l'estat
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
            # En cas d'error, assegurar-nos de que l'estat no es actualitzi
            if 'case' in locals():
                case.estat = "error"
                db.commit()
            return {
                "cas": data["cas"],
                "prediccions": [],
                "status": "error",
                "message": str(e)
            } 