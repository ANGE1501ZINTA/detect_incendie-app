from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
from model import CNN_Bottleneck

app = FastAPI(title="Fire Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # important pour GitHub Pages
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD MODEL
model = CNN_Bottleneck(bottleneck_dim=64)
model.load_state_dict(torch.load("fire_cnn_bottleneck.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

labels = ["Fire", "No Fire"]

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image only (JPG, PNG)")

    # Lecture et préparation de l'image
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Prédiction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)
    
    confidence_value = float(confidence)
    pred_class = int(pred)

    # Détermination du statut selon les seuils
    if confidence_value >= 0.85:
        status = "fire_detected" if pred_class == 0 else "no_fire"
        message = "🔥 Incendie détecté avec forte certitude !" if status == "fire_detected" else "✅ Aucun incendie détecté."
    elif confidence_value >= 0.5:
        status = "risk"
        message = "⚠️ Risque d'incendie détecté, vigilance recommandée."
    else:
        status = "no_fire"
        message = "✅ Probabilité faible d'incendie, situation normale."

    # Conversion image en base64 (optionnel, utile pour affichage frontend)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "success": True,
        "prediction": labels[pred_class],
        "confidence": round(confidence_value, 4),
        "status": status,
        "message": message,
        "image": f"data:image/png;base64,{img_str}"
    }
