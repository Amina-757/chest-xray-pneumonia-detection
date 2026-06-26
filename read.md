# Chest X-Ray Pneumonia Detection

Deep Learning project for pneumonia detection from chest X-ray images.

## Problem Statement
Pneumonia is a leading cause of death worldwide. This project uses Deep Learning to automatically detect pneumonia from chest X-ray images.

## Dataset
- **Source:** Kaggle Chest X-Ray Images (Pneumonia)
- **Classes:** NORMAL, PNEUMONIA

## Models
- **V1:** Baseline CNN
- **V2:** Transfer Learning with VGG16 + Fine-tuning

## API Endpoints
- `GET /` - Welcome
- `GET /health` - Health check
- `POST /predict` - Image prediction

## Docker
```bash
docker build -t chest-xray-pneumonia:v1 .
docker run -p 8000:8000 chest-xray-pneumonia:v1
kubectl apply -f kubernetes/
-----------------------
