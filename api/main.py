from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import json
import os
from typing import List, Optional, Dict
import logging
from PIL import Image
import io

from ..models.architectures.hurricane_model import create_model
from ..data.preprocessing.data_processor import DataProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Hurricane Prediction API",
    description="API for predicting hurricane formation and tracking using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
with open('models/saved/config.json', 'r') as f:
    config = json.load(f)

# Initialize model and move to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(config).to(device)

# Load the best checkpoint
checkpoint_path = os.path.join(config['output_dir'], 'best_model.pt')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info(f"Loaded model from {checkpoint_path}")
else:
    logging.warning(f"No checkpoint found at {checkpoint_path}")

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    category: int
    category_probability: float
    track_coordinates: Optional[List[Dict[str, float]]] = None
    confidence_score: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_hurricane(
    file: UploadFile = File(...),
    predict_track: bool = True
):
    """
    Predict hurricane category and track from satellite imagery
    
    Args:
        file: Satellite image file (JPEG, PNG, or NetCDF)
        predict_track: Whether to predict track coordinates
    
    Returns:
        PredictionResponse containing category, probability, and track coordinates
    """
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to expected dimensions
        image = image.resize((256, 256))  # Adjust size based on your model's requirements
        
        # Convert to tensor and normalize
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # Convert to CHW format
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            if predict_track:
                category_pred, track_pred = model(image_tensor)
            else:
                category_pred = model(image_tensor)
        
        # Process predictions
        category_probs = torch.softmax(category_pred, dim=1)
        category = torch.argmax(category_probs).item()
        category_probability = category_probs[0, category].item()
        
        # Prepare response
        response = {
            "category": category,
            "category_probability": category_probability,
            "confidence_score": category_probability
        }
        
        if predict_track:
            # Convert track predictions to list of coordinates
            track_coords = track_pred[0].cpu().numpy()
            response["track_coordinates"] = [
                {"latitude": float(coord[0]), "longitude": float(coord[1])}
                for coord in track_coords
            ]
        
        return PredictionResponse(**response)
    
    except Exception as e:
        logging.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": os.path.exists(checkpoint_path)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 