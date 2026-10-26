import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Datos simulados con 2 variables numéricas
X = np.array([[10.0, 1.2], [42.5, 1.75], [5.0, 0.5], [80.0, 3.0]])
y = np.array([0, 1, 0, 1])

model = LogisticRegression()
model.fit(X, y)

# Guardar en la ruta esperada por la arquitectura
joblib.dump(model, "model_artifacts/model.joblib")
print("Modelo guardado en model_artifacts/model.joblib")