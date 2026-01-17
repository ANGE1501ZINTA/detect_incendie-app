import streamlit as st 
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import CNN_Bottleneck

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="Fire Detection AI",
    page_icon="🔥",
    layout="centered"
)

st.title("🔥 Fire Detection using Deep Learning")
st.write("Application de détection automatique d'incendie à partir d'images.")

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    model = CNN_Bottleneck(bottleneck_dim=64)
    model.load_state_dict(
        torch.load("fire_cnn_bottleneck.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

labels = ["🔥 Fire", "✅ No Fire"]

# -----------------------
# IMAGE UPLOAD
# -----------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image analysée")


    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred = pred.item()

    st.subheader("🧠 Résultat de la prédiction")
    st.write(f"**Classe prédite : {labels[pred]}**")
    st.write(f"**Confiance : {confidence:.2f}**")

    # Barre de probabilité
    st.progress(confidence)

    # Message intelligent
    if confidence < 0.6:
        st.warning("⚠️ Le modèle est incertain sur cette image.")
    elif pred == 0:
        st.success("🔥 Incendie détecté !")
    else:
        st.success("✅ Aucun incendie détecté.")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Projet Deep Learning — Détection d'incendie | Master 2 Data Science")
