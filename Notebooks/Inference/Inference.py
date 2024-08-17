import pandas as pd
import pickle

# Cargar el pipeline entrenado desde un archivo
with open('../Models/rf_classifier_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Cargar nuevos datos para inferencia
new_data = pd.read_csv('../../Data/Raw/melb_data.csv')

# Seleccionar solo las columnas numéricas y eliminar las filas con valores NaN
df_numerico = new_data.select_dtypes(include=['number']).dropna(axis=1)
df_numerico = df_numerico.drop(columns=['Longtitude', 'Lattitude', 'Price'])

# Eliminar columnas 'Longtitude' y 'Lattitude' si están presentes
if 'Longtitude' in df_numerico.columns:
    df_numerico = df_numerico.drop(columns=['Longtitude'])
if 'Lattitude' in df_numerico.columns:
    df_numerico = df_numerico.drop(columns=['Lattitude'])

# Asegurar que las características seleccionadas sean las mismas que durante el entrenamiento
# En este caso, simplemente utilizamos el pipeline que ya se encargará de la selección
df_numerico = df_numerico[pipeline.named_steps['selector'].get_support(indices=True)]

# Realizar predicciones usando el pipeline cargado
predictions = pipeline.predict(df_numerico)

# Mostrar predicciones
print("Predicciones:", predictions)