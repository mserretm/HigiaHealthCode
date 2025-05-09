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
            Case.us_estatentrenament == 0,
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
            if case.us_estatentrenament != 0:
                return {
                    "cas": data["cas"],
                    "prediccions": [],
                    "status": "info",
                    "message": f"El cas {data['cas']} ja ha estat processat (estat: {case.us_estatentrenament})"
                }

            # Realitzar la predicció
            result = self.engine.predict(data)
            
            # Actualitzar l'estat i les prediccions en la base de dades
            case.us_estatentrenament = 1
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
                case.us_estatentrenament = 2  # Marcar com a error
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

    async def process_train(self, data: dict, db: Session) -> dict:
        """
        Procesa un caso para entrenamiento.
        """
        try:
            # Verificar que el cas existeix en la base de dades
            case = db.query(Case).filter(Case.cas == data["cas"]).first()
            if not case:
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {data['cas']}")
            
            # Realitzar l'entrenament
            await self.engine.train_incremental(data, db)
            
            # Actualizar el estado del caso
            case.estat = "entrenat"
            db.commit()
            
            return {
                "cas": data["cas"],
                "status": "success",
                "message": "Entrenament completat correctament"
            }
        except Exception as e:
            logger.error(f"Error en l'entrenament del cas {data['cas']}: {str(e)}")
            if 'case' in locals():
                case.estat = "error"
                db.commit()
            raise

    async def process_validate(self, case_data: dict, db: Session) -> dict:
        """
        Procesa un caso para validación.
        """
        try:
            logger.info(f"Iniciant validació del cas {case_data.get('cas', 'N/A')}...")
            
            # Verificar que el cas existeix en la base de dades
            case = db.query(Case).filter(Case.cas == case_data["cas"]).first()
            if not case:
                logger.error(f"No s'ha trobat el cas {case_data['cas']} a la base de dades.")
                raise HTTPException(status_code=404, detail=f"No s'ha trobat el cas {case_data['cas']}")
            
            logger.info(f"Cas {case_data['cas']} trobat a la base de dades. Iniciant validació amb el model...")
            
            # Validar el modelo con el caso
            result = self.engine.validate_model(case_data, db)
            
            # Actualizar el estado del caso
            case.estat = "validat"
            db.commit()
            logger.info(f"Estat del cas {case_data['cas']} actualitzat a 'validat'.")
            
            return result
        except Exception as e:
            logger.error(f"Error en la validació del cas {case_data.get('cas', 'N/A')}: {str(e)}")
            if 'case' in locals():
                case.estat = "error"
                db.commit()
                logger.info(f"Estat del cas {case_data.get('cas', 'N/A')} actualitzat a 'error'.")
            raise

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