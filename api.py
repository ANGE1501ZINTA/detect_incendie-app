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

# 🔴 CORRECTION CORS CRITIQUE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # OK
    allow_credentials=False,    # ⬅️ OBLIGATOIRE
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
        raise HTTPException(status_code=400, detail="Image only")

    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return {
        "success": True,
        "class": labels[int(pred)],
        "confidence": float(confidence)
    }
