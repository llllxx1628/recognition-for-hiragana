from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from typing import List
from model import HiraganaRecognitionNet

app = FastAPI(title="Hiragana Character Recognition API",
              description="API for predicting Hiragana characters using a pretrained model",
              version="1.0")

# Load the pretrained model
model_path = "best_model.pth"
model = HiraganaRecognitionNet(num_classes=49)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = Compose([
    ToTensor(),
    Normalize(mean=(0.5,), std=(0.5,))
])

# API input and output schemas
class PredictionRequest(BaseModel):
    filenames: List[str]

class PredictionResponse(BaseModel):
    filename: str
    prediction: int

@app.post("/predict", response_model=List[PredictionResponse])
async def predict(file: List[UploadFile] = File(...)):
    """
    Predict Hiragana characters from uploaded image files.
    
    Args:
        file (List[UploadFile]): List of image files to be uploaded.

    Returns:
        List[PredictionResponse]: List of predictions for each image.
    """
    predictions = []
    for uploaded_file in file:
        try:
            image = Image.open(uploaded_file.file).convert('L')  
            input_tensor = transform(image).unsqueeze(0).to(device)  
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted_class = torch.max(outputs, 1)
                predictions.append(
                    PredictionResponse(
                        filename=uploaded_file.filename, prediction=predicted_class.item()
                    )
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file {uploaded_file.filename}: {str(e)}"
            )

    return predictions

# API endpoint for health check
@app.get("/health")
def health_check():

    return JSONResponse(content={"status": "API is healthy and running"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
