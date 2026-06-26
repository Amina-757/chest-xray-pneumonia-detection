
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from datetime import datetime

app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="API for pneumonia detection from chest X-ray images",
    version="1.0.0"
)

# Load model
import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
from datetime import datetime

app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="API for pneumonia detection from chest X-ray images",
    version="1.0.0"
)

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', '/app/checkpoints/improved_cnn_final.keras')
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded successfully from {MODEL_PATH}")

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    probability: float
    model_version: str
    timestamp: str

@app.get("/")
def root():
    return {
        "message": "Welcome to Chest X-Ray Pneumonia Detection API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict"],
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": "V2-Improved-CNN",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prob = float(model.predict(img_array, verbose=0)[0][0])
    pred = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    conf = prob if prob > 0.5 else 1 - prob
    
    return PredictionResponse(
        filename=file.filename,
        prediction=pred,
        confidence=round(conf, 4),
        probability=round(prob, 4),
        model_version="V2-Improved-CNN",
        timestamp=datetime.now().isoformat()
    )
