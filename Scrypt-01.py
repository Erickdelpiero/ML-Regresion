import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Explorar las dimensiones y las primeras filas del dataset
print(np.shape(data))
print(data.head())

# Resumen de los datos distribuidos
print(data.describe())


# Seleccionar una característica y la variable objetivo
X = data[['rm']]  # promedio de habitaciones por vivienda
y = data['medv']  # valor medio de las casas ocupadas por sus propietarios en $1000's

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir y calcular el error cuadrático medio
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Error Cuadrático Medio: {mse}")

# Mostrar los coeficientes del modelo
print(f"Coeficiente de la pendiente: {model.coef_[0]}, Intercepto: {model.intercept_}")
