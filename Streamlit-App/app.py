import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os


# Configuración de la app
st.set_page_config(page_title="Detección de Fachadas", layout="centered")

st.title("🏛️ Detección de elementos en Fachadas")
st.subheader("Detección de Puertas, Ventanas, Balcones y Barandas")
st.write("Sube una imagen y el modelo YOLO detectará los elementos arquitectónicos.")


# Cargar el modelo
@st.cache_resource
def load_model():
    model = YOLO("D:/Documentos/3RO/1ER Semestre/RN/Object-detection/YOLOV8/Resultados Yolo/yolo_final/weights/best.pt")  # Asegúrate de tener best.pt en la misma carpeta
    return model

model = load_model()


conf_threshold = st.slider(
    "Umbral de confianza", 
    min_value=0.1, 
    max_value=0.9, 
    value=0.5, 
    step=0.05
)


# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

if uploaded_file is not None:
    # Mostrar imagen original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Guardar temporalmente la imagen
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        image.save(tmp_path)

    
    # Ejecutar yolo
    st.subheader("Detecciones")
    results = model.predict(tmp_path, conf=conf_threshold)

    # Mostrar imagen con detecciones
    result_img = results[0].plot()  # Imagen con cajas dibujadas
    st.image(result_img, caption="Resultado YOLO", use_column_width=True)

    
    # Mostrar detalles por clases
    st.subheader("📊 Conteo por clase")

    names = model.names
    counts = {}

    for box in results[0].boxes:
        cls = int(box.cls[0])
        cls_name = names[cls]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    if len(counts) == 0:
        st.write("No se detectaron objetos.")
    else:
        for cls_name, count in counts.items():
            st.write(f"**{cls_name}:** {count}")

    # Eliminar archivo temporal
    os.remove(tmp_path)
