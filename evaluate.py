import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


# Crear carpeta de resultados si no existe
os.makedirs("Results", exist_ok=True)

# Cargar modelo y datos de prueba guardados durante el entrenamiento
model = joblib.load("Model/model.pkl")
X_test = joblib.load("Model/X_test.pkl")
y_test = joblib.load("Model/y_test.pkl")

# Generar predicciones
y_pred = model.predict(X_test)

# Crear reporte de clasificación
report = classification_report(y_test, y_pred)

# Guardar reporte en el archivo de métricas
with open("Results/metrics.txt", "a") as f:
    f.write("\nClassification Report:\n")
    f.write(report)

# Crear y guardar matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("Results/model_results.png")
plt.close()

print("Evaluación finalizada correctamente.")
