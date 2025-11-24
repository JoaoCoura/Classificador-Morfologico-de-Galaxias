import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Galaxy Classifier",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background: black !important;
        }}
        h1, h2, h3, h4, h5, h6, p, label {{
            color: white !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
<style>
.divider {
    border-left: 2px solid white;
    height: 100%;
    margin: 0 20px;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH_RESNET = "arquiteturaResNet50.keras"
MODEL_PATH_AUTHOR = "arquiteturaPropria.keras"

@st.cache_resource
def load_resnet():
    return load_model(MODEL_PATH_RESNET)

@st.cache_resource
def load_author():
    return load_model(MODEL_PATH_AUTHOR)

class_names = [
    "Disturbed Galaxy", "Merging Galaxy", "Round Smooth Galaxy",
    "In-between Round Smooth Galaxy", "Cigar Shaped Smooth Galaxy",
    "Barred Spiral Galaxy", "Unbarred Tight Spiral Galaxy",
    "Unbarred Loose Spiral Galaxy", "Edge-on Galaxy (No Bulge)",
    "Edge-on Galaxy (With Bulge)"
]

def predict_resnet(img):
    IMG_SIZE = 256
    model = load_resnet()

    arr = np.array(img)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(arr, axis=0).astype("float32")
    arr = preprocess_input(arr)

    pred = model.predict(arr)[0]
    class_idx = np.argmax(pred)
    conf = pred[class_idx]

    return class_idx, conf, pred


def predict_autoral(img):
    IMG_SIZE = 128
    model = load_author()

    arr = np.array(img)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(arr, axis=0).astype("float32")
    arr /= 255.0

    pred = model.predict(arr)[0]
    class_idx = np.argmax(pred)
    conf = pred[class_idx]

    return class_idx, conf, pred


st.title("🌌 Classificador de Galáxias - Ensemble")
st.write("Envie uma imagem para ser classificada por dois modelos e combinados.")

uploaded = st.file_uploader("Selecione uma imagem", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagem enviada", width='stretch')

    if st.button("Classificar Galáxia"):
        idx_res, conf_res, pred_res = predict_resnet(img)
        idx_aut, conf_aut, pred_aut = predict_autoral(img)

        pred_mean = (pred_res + pred_aut) / 2
        idx_mean = np.argmax(pred_mean)
        conf_mean = pred_mean[idx_mean]

        col1, col_div, col2 = st.columns([1, 0.1, 1])

        with col1:
            st.subheader("Modelo ResNet-50")
            st.write(f"**Classe:** {class_names[idx_res]}")
            st.write(f"**Confiança:** {conf_res*100:.2f}%")
            st.write("**Distribuição das Confianças:**")
            for i, p in enumerate(pred_res):
                st.write(f"{class_names[i]}: {p*100:.2f}%")

        with col_div:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        with col2:
            st.subheader("Modelo Autoral")
            st.write(f"**Classe:** {class_names[idx_aut]}")
            st.write(f"**Confiança:** {conf_aut*100:.2f}%")
            st.write("**Distribuição das Confianças:**")
            for i, p in enumerate(pred_aut):
                st.write(f"{class_names[i]}: {p*100:.2f}%")

        st.subheader("Resultado Final (Ensemble por Média)")
        st.write(f"**Classe:** {class_names[idx_mean]}")
        st.write(f"**Confiança:** {conf_mean*100:.2f}%")

        st.subheader("Distribuição das Confianças (Ensemble)")
        for i, p in enumerate(pred_mean):
            st.write(f"{class_names[i]}: {p*100:.2f}%")
