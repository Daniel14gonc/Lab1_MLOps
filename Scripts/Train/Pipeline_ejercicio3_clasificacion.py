import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Cargar datos
data = pd.read_csv('../../Data/Raw/melb_data.csv')

# Crear categoría de precio
data['categoria_precio'] = pd.qcut(data['Price'], q=4, labels=['Bajo', 'Medio Bajo', 'Medio Alto', 'Alto'])

# Seleccionar solo las columnas numéricas y eliminar las filas con valores NaN
df_numerico = data.select_dtypes(include=['number']).dropna(axis=1)

# Eliminar columnas 'Longtitude' y 'Lattitude'
df_numerico = df_numerico.drop(columns=['Longtitude', 'Lattitude'])

# Definir las características (X) y la variable objetivo (y)
X = df_numerico.drop(columns=['Price'])
y = data['categoria_precio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Escalar los datos
    ('selector', SelectKBest(score_func=f_classif, k=7)),  # Selección de características
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Modelo de clasificación
])

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

# Hacer predicciones
y_pred = pipeline.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Mostrar el reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Características seleccionadas
selector = pipeline.named_steps['selector']
mask = selector.get_support()  # Array booleano de las características seleccionadas
selected_features = X.columns[mask]
print("Características seleccionadas:", selected_features)

with open('../Models/rf_classifier_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)