import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os


def train_system():
    data_path = "../data/processed/data.csv"

    if not os.path.exists(data_path):
        print("Error: No se encuentra el CSV en data/processed/")
        return

    df = pd.read_csv(data_path, sep=";")

    print("Columnas:", df.columns.tolist())
    print("Muestras:", len(df))

    print("\nConteo por sílaba:")
    print(df["Silaba"].value_counts())

    print("\nConteo por género:")
    print(df["Genero"].value_counts())

    # Género: principalmente F0
    X_genero = df[["F0"]]
    y_genero = df["Genero"]

    scaler_genero = StandardScaler()
    X_genero_scaled = scaler_genero.fit_transform(X_genero)

    modelo_genero = KNeighborsClassifier(n_neighbors=5)
    modelo_genero.fit(X_genero_scaled, y_genero)

    # Sílaba: características de vocal
    X_silaba = df[[
        "F1",
        "F2",
        "E_baja",
        "E_media",
        "Fdom",
        "Centroide",
        "Relacion_E",
        "Relacion_F"
    ]]

    y_silaba = df["Silaba"]

    scaler_silaba = StandardScaler()
    X_silaba_scaled = scaler_silaba.fit_transform(X_silaba)

    modelo_silaba = SVC(
        kernel="rbf",
        C=20,
        gamma="scale"
    )

    modelo_silaba.fit(X_silaba_scaled, y_silaba)

    os.makedirs("models", exist_ok=True)

    joblib.dump(scaler_genero, "models/scaler_genero.pkl")
    joblib.dump(modelo_genero, "models/knn_genero.pkl")

    joblib.dump(scaler_silaba, "models/scaler_silaba.pkl")
    joblib.dump(modelo_silaba, "models/knn_silaba.pkl")

    print("\nModelos guardados correctamente.")


if __name__ == "__main__":
    train_system()