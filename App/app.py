import joblib
import gradio as gr
import numpy as np


# Cargar el modelo entrenado
model = joblib.load("Model/model.pkl")


# Función de predicción
def predict(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]

    classes = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    return classes[prediction]


# Interfaz web con Gradio
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal length"),
        gr.Number(label="Sepal width"),
        gr.Number(label="Petal length"),
        gr.Number(label="Petal width"),
    ],
    outputs=gr.Textbox(label="Predicción"),
    title="Iris Classification",
    description="Aplicación desplegada con CI/CD usando GitHub Actions y Hugging Face Spaces."
)


if __name__ == "__main__":
    demo.launch()
