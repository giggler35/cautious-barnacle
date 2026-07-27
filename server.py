"""
server.py

A small FastAPI server that runs the trained skin classifier and
requires an API key for access -- so only your team (people you give
the key to) can call it.

Usage:
    pip install fastapi uvicorn python-multipart timm torch torchvision pillow
    python server.py

Then share:
  - The ngrok URL (changes each time you restart ngrok, unless you're on
    a paid plan)
  - The API_KEY value below (generate your own, don't use the placeholder)

Your team calls it like:
    curl -X POST "https://your-ngrok-url.ngrok-free.app/predict" \
         -H "X-API-Key: YOUR_KEY_HERE" \
         -F "file=@photo.jpg"
"""

import io
import os

import timm
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image
from torchvision import transforms

from disease_codes import NAME_TO_CODE

# ---------------------------------------------------------------------
# CHANGE THIS -- generate your own random key, don't use this placeholder.
# A simple way: run this in Python once and paste the result here:
#   import secrets; print(secrets.token_urlsafe(32))
# ---------------------------------------------------------------------
API_KEY = "CHANGE_ME_TO_A_RANDOM_STRING"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "skin_effnetv2_b2.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="RashAI Inference Server")

# ---------------------------------------------------------------------
# Load model once at startup, not per-request
# ---------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
CLASS_NAMES = checkpoint["class_names"]
IMG_SIZE = checkpoint["img_size"]
DROPOUT = checkpoint.get("dropout", 0.4)
TIMM_MODEL_NAME = checkpoint.get("timm_model_name", "tf_efficientnetv2_b2")

model = timm.create_model(
    TIMM_MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES), drop_rate=DROPOUT
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print(f"Model loaded. {len(CLASS_NAMES)} classes. Using {DEVICE}.")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def check_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def health_check():
    return {"status": "ok", "model": TIMM_MODEL_NAME, "num_classes": len(CLASS_NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(None)):
    check_api_key(x_api_key)

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    x = val_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    top_idx = torch.argmax(probs).item()
    top_name = CLASS_NAMES[top_idx]

    return PlainTextResponse(NAME_TO_CODE[top_name])


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
