from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import logging
from pathlib import Path
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
PORT = int(os.getenv("PORT", "8000"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


base_path = Path(__file__).parent.parent
logger.info(f"Looking for model files in: {base_path}")

try:
    logger.info("Loading vectorizer...")
    vectorizer = joblib.load(base_path / "vectorizer.pkl")
    logger.info("Vectorizer loaded successfully")
    
    logger.info("Loading language classifier...")
    model = joblib.load(base_path / "xgb_model.pkl")
    logger.info("Language classifier loaded successfully")
    
    logger.info("Loading label encoder...")
    label_encoder = joblib.load(base_path / "label_encoder.pkl")
    logger.info("Label encoder loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {str(e)}")
    logger.error(f"Current directory contents: {list(base_path.glob('*.pkl'))}")
    raise RuntimeError(f"Failed to load required models: {str(e)}")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise RuntimeError(f"Failed to load required models: {str(e)}")

class CodeSnippet(BaseModel):
    code: str

@app.get("/")
def read_root():
    return {
        "message": "Programming Language Detector API",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_language(code_snippet: CodeSnippet):
    try:
        if not code_snippet.code.strip():
            raise HTTPException(status_code=400, detail="Code snippet cannot be empty")
            
        logger.info("Vectorizing input code...")
        code_vectorized = vectorizer.transform([code_snippet.code])
        
        logger.info("Making prediction...")
        prediction = model.predict(code_vectorized)
        
        logger.info("Converting prediction to language name...")
        language = label_encoder.inverse_transform(prediction)[0]
        
        logger.info(f"Detected language: {language}")
        return {
            "language": language,
            "confidence": "high"
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 