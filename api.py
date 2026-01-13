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

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# LOAD MODEL
# -----------------------
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

labels = ["Fire", "No Fire"]

# -----------------------
# ROUTES
# -----------------------
@app.get("/")
async def home():
    return {
        "message": "Fire Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "/predict": "POST - Upload image for fire detection",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health():
    """Vérifier l'état de santé de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Endpoint de prédiction pour la détection d'incendie.
    """
    try:
        # Vérifier le type de fichier
        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Le fichier doit être une image (JPG, PNG)"
            )
        
        # Lire et convertir l'image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Préparer l'image pour le modèle
        img_tensor = transform(img).unsqueeze(0)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        
        confidence_value = float(confidence.item())
        prediction = int(pred.item())
        
        # Statut + messages nuancés
        if prediction == 0:  # Fire
            if confidence_value >= 0.8:
                status = "fire_detected"
                message = "🔥 Incendie détecté avec forte certitude ! Évacuation immédiate recommandée."
            elif confidence_value >= 0.6:
                status = "fire_detected"
                message = "🔥 Incendie détecté. Vérification visuelle conseillée."
            elif confidence_value >= 0.5:
                status = "fire_detected"
                message = "⚠️ Risque élevé d'incendie. Fumée ou flammes possibles. Inspection immédiate requise."
            else:
                status = "uncertain"
                message = "⚠️ Situation ambiguë. Présence de fumée ou conditions suspectes. Surveillance recommandée."
        else:  # No Fire
            if confidence_value >= 0.8:
                status = "no_fire"
                message = "✅ Aucun incendie détecté. Situation normale."
            elif confidence_value >= 0.6:
                status = "no_fire"
                message = "✅ Pas d'incendie apparent, mais restez vigilant."
            else:
                status = "uncertain"
                message = "⚠️ Le modèle est incertain. Conditions d'éclairage ou image floue possible."
        
        # Convertir l'image en base64 pour l'affichage
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "prediction": {
                "class": labels[prediction],
                "class_id": prediction,
                "confidence": round(confidence_value, 4),
                "status": status,
                "message": message
            },
            "probabilities": {
                "fire": round(float(probs[0][0].item()), 4),
                "no_fire": round(float(probs[0][1].item()), 4)
            },
            "image": f"data:image/png;base64,{img_str}"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )
