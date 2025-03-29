from sqlalchemy import text
from sqlalchemy.orm import Session

def get_pending_records(db: Session):
    """
    Obté els registres pendents d'entrenament.
    """
    query = text("""
        SELECT 
            cas,
            dx_revisat,
            edat,
            genere,
            c_alta,
            periode,
            servei,
            motiuingres,
            malaltiaactual,
            exploracio,
            provescomplementariesing,
            provescomplementaries,
            evolucio,
            antecedents,
            cursclinic
        FROM public.f_cmbd_ha_deeplearning
        WHERE us_registre = 'T' 
        AND us_estatentrenament = 0 limit 10
    """)
    return db.execute(query).fetchall()

def update_training_status(db: Session, cas: str, status: str):
    """
    Actualitza l'estat d'entrenament d'un registre.
    """
    status_value = 1 if status == "Entrenat" else 2 if status == "Error" else 0
    
    query = text("""
        UPDATE public.f_cmbd_ha_deeplearning
        SET us_estatentrenament = :status
        WHERE cas = :cas
    """)
    db.execute(query, {"cas": cas, "status": status_value})
    db.commit() 