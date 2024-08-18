import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos
data = pd.read_csv('../../Data/Raw/melb_data.csv')

# Seleccionar solo las columnas numéricas y eliminar las filas con valores NaN
df_numerico = data.select_dtypes(include=['number']).dropna(axis=1)

# Eliminar columnas 'Longtitude' y 'Lattitude'
df_numerico = df_numerico.drop(columns=['Longtitude', 'Lattitude'])

# Definir las características (X) y la variable objetivo (y)
X = df_numerico.drop(columns=['Price'])
y = df_numerico['Price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Escalar los datos
    ('selector', SelectKBest(score_func=f_classif, k=5)),  # Selección de características
    ('regressor', LinearRegression())  # Modelo de regresión lineal
])

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

# Predicciones
y_pred = pipeline.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Características seleccionadas
selector = pipeline.named_steps['selector']
mask = selector.get_support()  # Array booleano de las características seleccionadas
selected_features = X.columns[mask]
print("Características seleccionadas:", selected_features)