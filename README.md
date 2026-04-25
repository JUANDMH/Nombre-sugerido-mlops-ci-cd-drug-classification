# Proyecto CI/CD para Machine Learning

Estudiante: JUAN MARIN  
Grupo: Grupo 1  

## Descripción del proyecto

Este proyecto implementa un flujo de integración continua y despliegue continuo para un modelo de Machine Learning.

El objetivo es automatizar el ciclo completo del modelo, incluyendo:

1. Instalación de dependencias.
2. Formateo del código Python.
3. Entrenamiento del modelo.
4. Evaluación del modelo.
5. Generación de métricas y matriz de confusión.
6. Publicación de resultados con CML.
7. Despliegue de una aplicación en Hugging Face Spaces.

## Dataset utilizado

Para este ejercicio se utilizó el dataset Iris incluido en la librería scikit-learn.

Este dataset contiene mediciones de flores a partir de cuatro variables:

- Sepal length.
- Sepal width.
- Petal length.
- Petal width.

La variable objetivo corresponde a la especie de la flor:

- Setosa.
- Versicolor.
- Virginica.

## Modelo utilizado

Se utilizó un modelo Random Forest Classifier de la librería scikit-learn.

El modelo se entrena en el archivo `train.py` y posteriormente se evalúa en el archivo `evaluate.py`.

## Estructura del proyecto

```text
.
├── App/
│   └── app.py
├── Model/
├── Results/
├── train.py
├── evaluate.py
├── requirements.txt
├── Makefile
└── README.md
