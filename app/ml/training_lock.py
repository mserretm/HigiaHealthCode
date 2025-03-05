# app/ml/training_lock.py

import threading

# Crear un lock global per l'entrenament
training_lock = threading.Lock()
