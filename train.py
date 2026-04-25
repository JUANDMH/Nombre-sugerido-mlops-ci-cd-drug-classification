import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Crear carpetas necesarias
os.makedirs("Model", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# Cargar dataset de ejemplo
data = load_iris()

# Crear variables predictoras y variable objetivo
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Crear modelo de clasificación
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entrenar modelo
model.fit(X_train, y_train)

# Generar predicciones
y_pred = model.predict(X_test)

# Calcular métrica
accuracy = accuracy_score(y_test, y_pred)

# Guardar modelo y datos de prueba
joblib.dump(model, "Model/model.pkl")
joblib.dump(X_test, "Model/X_test.pkl")
joblib.dump(y_test, "Model/y_test.pkl")

# Guardar métrica en archivo de resultados
with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")

print("Modelo entrenado correctamente.")
print(f"Accuracy: {accuracy:.4f}")
