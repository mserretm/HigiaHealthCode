from sqlalchemy import Column, Integer, String, Float, ARRAY, DateTime, Sequence, Text
from app.db.base_class import Base

class DeepLearningEvaluate(Base):
    __tablename__ = 'f_deeplearning_evaluate'
    
    id = Column(Integer, Sequence('f_deeplearning_evaluate_id_seq'), primary_key=True)
    experimento_id = Column(String(100), nullable=False)
    caso_id = Column(String(100), nullable=False)
    
    # Códigos y orden
    codis_reals = Column(ARRAY(Text), nullable=True)
    codis_predits_top15 = Column(ARRAY(Text), nullable=True)
    probs_predits_top15 = Column(ARRAY(Float), nullable=True)
    codis_predits_confianza_top15 = Column(ARRAY(Text), nullable=True)
    ordre_real = Column(ARRAY(Text), nullable=True)
    ordre_predit = Column(ARRAY(Text), nullable=True)
    
    # Métricas de clasificación
    accuracy = Column(Float, nullable=True)
    precisi = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Métricas de orden
    order_accuracy = Column(Float, nullable=True)
    kendall_tau = Column(Float, nullable=True)
    
    # Pérdidas
    code_loss = Column(Float, nullable=True)
    order_loss = Column(Float, nullable=True)
    
    # Contadores
    num_codis_reals = Column(Integer, nullable=True)
    num_codis_predits_top15 = Column(Integer, nullable=True)
    num_codis_predits_confianza_top15 = Column(Integer, nullable=True)
    
    # Metainformación opcional
    ver_modelo = Column(String(50), nullable=True)
    set_validacion = Column(String(100), nullable=True)
    ts_inici = Column(DateTime, nullable=True)
    ts_final = Column(DateTime, nullable=True) 