# Importar las librerías necesarias
from sklearn.linear_model import LinearRegression
import numpy as np

# Datos de ejemplo
# X representa las horas de estudio
X = np.array([[5], [15], [25], [35], [45], [55]])
# y representa las calificaciones obtenidas
y = np.array([5, 20, 14, 32, 22, 38])

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos proporcionados
model.fit(X, y)

a=input('Cantidad de horas de estudio: ')
a=int(a)
# Predecir la calificación para un nuevo dato de entrada
X_new = np.array([[a]])  # Supongamos que un estudiante estudia 50 horas
prediction = model.predict(X_new)
print(f"Predicción de calificación: {prediction[0]}")
