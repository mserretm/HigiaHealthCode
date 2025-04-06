from sqlalchemy import Column, Integer, String, DateTime, Float, Sequence, Date, Time
from app.db.base_class import Base

class DeepLearningTrain(Base):
    __tablename__ = 'f_deeplearning_train'
    
    id = Column(Integer, Sequence('f_deeplearning_train_id_seq'), primary_key=True)
    cas = Column(String(25), nullable=False)
    data_entrenament = Column(Date, nullable=False)
    hora_inici = Column(String(8), nullable=False)
    hora_fi = Column(String(8), nullable=False)
    durada = Column(String(8), nullable=False)  # Duración en segundos
    lr_inicial = Column(Float, nullable=False)
    decay = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    max_epochs = Column(Integer, nullable=False)
    early_stopping_patience = Column(Integer, nullable=False)
    max_len_input = Column(Integer, nullable=False)
    umbral_confianza_prediccio = Column(Float, nullable=False)
    epochs = Column(Integer, nullable=False, default=0)
    lr_final = Column(Float, nullable=False, default=0)
    loss_total_final = Column(Float, nullable=False, default=0)
    loss_code_final = Column(Float, nullable=False, default=0)
    loss_order_final = Column(Float, nullable=False, default=0) 
    
    