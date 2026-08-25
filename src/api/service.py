from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil
from contextlib import asynccontextmanager

from src.inference.whisper_engine import WhisperEngine
from src.data.text_normalizer import normalize_text
from src.inference.evaluator import evaluate_metrics

# Initialize globally to keep model in memory
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = WhisperEngine()
    yield

app = FastAPI(title="Stateless ASR Evaluation API", lifespan=lifespan)

@app.post("/evaluate")
async def evaluate_audio(
    file: UploadFile = File(...),
    reference_text: str = Form(None)
):
    """
    Stateless evaluation:
    1. Upload a WAV file
    2. Transcribe it
    3. If reference_text provided, evaluate WER/CER.
    """
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
        
    try:
        # Save temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Transcribe
        predicted_text = engine.transcribe(temp_path)
        normalized_pred = normalize_text(predicted_text)
        
        response = {
            "predicted_text_raw": predicted_text,
            "predicted_text_normalized": normalized_pred
        }
        
        if reference_text:
            normalized_ref = normalize_text(reference_text)
            metrics = evaluate_metrics(normalized_ref, normalized_pred)
            response["metrics"] = metrics
            
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
