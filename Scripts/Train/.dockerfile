# Utilizar una imagen base de Python 3.8 (puedes ajustar la versión si lo prefieres)
FROM python:3.8-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos y el script de Python al contenedor
COPY requirements.txt requirements.txt
COPY . .

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar el script de Python
CMD ["python", "Pipeline_ejercicio3_clasificacion.py"]
