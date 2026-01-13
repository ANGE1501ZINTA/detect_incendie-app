from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
from model import CNN_Bottleneck

# -----------------------
# INIT FASTAPI
# -----------------------
app = FastAPI(
    title="Fire Detection API",
    description="API de détection d'incendie par Deep Learning",
    version="1.0.0"
)

# -----------------------
# CORS (GitHub Pages → Render)
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ange1501zinta.github.io"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# LOAD MODEL
# -----------------------
def load_model():
    try:
        model = CNN_Bottleneck(bottleneck_dim=64)
        model.load_state_dict(
            torch.load("fire_cnn_bottleneck.pth", map_location="cpu")
        )
        model.eval()
        return model
    except Exception as e:
        print("❌ ERREUR CHARGEMENT MODÈLE :", e)
        return None

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

labels = ["Fire", "No Fire"]

# -----------------------
# ROUTES
# -----------------------
@app.get("/")
def home():
    return {
        "message": "Fire Detection API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image"
        )

    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        confidence_value = float(confidence.item())
        prediction = int(pred.item())

        if prediction == 0:
            status = "fire_detected" if confidence_value >= 0.6 else "uncertain"
            message = "🔥 Incendie détecté" if status == "fire_detected" else "⚠️ Situation incertaine"
        else:
            status = "no_fire" if confidence_value >= 0.6 else "uncertain"
            message = "✅ Aucun incendie détecté" if status == "no_fire" else "⚠️ Situation incertaine"

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "prediction": {
                "class": labels[prediction],
                "confidence": round(confidence_value, 4),
                "status": status,
                "message": message
            },
            "probabilities": {
                "fire": round(float(probs[0][0]), 4),
                "no_fire": round(float(probs[0][1]), 4)
            },
            "image": f"data:image/png;base64,{img_str}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
