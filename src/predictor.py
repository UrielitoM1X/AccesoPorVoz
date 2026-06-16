import joblib
import numpy as np
import os


class AudioPredictor:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.scaler_genero = joblib.load(
            os.path.join(base_dir, "models", "scaler_genero.pkl")
        )

        self.modelo_genero = joblib.load(
            os.path.join(base_dir, "models", "knn_genero.pkl")
        )

        print("✅ Modelo de género cargado correctamente.")

    def predict_genero(self, f0):
        X_genero = np.array([[f0]], dtype=float)

        X_genero_scaled = self.scaler_genero.transform(X_genero)

        genero = self.modelo_genero.predict(X_genero_scaled)[0]

        return genero