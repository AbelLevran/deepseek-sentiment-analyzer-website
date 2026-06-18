import os
import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, constr
from typing import List

from preprocess.cleaner import preprocess_text

router = APIRouter()

# Load models safely
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "../model/model.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", os.path.join(os.path.dirname(__file__), "../model/vectorizer.pkl"))

model = None
vectorizer = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        print("Warning: Model or vectorizer not found.")
except Exception as e:
    print(f"Error loading model: {e}")

class PredictRequest(BaseModel):
    text: constr(min_length=1, max_length=5000)

class PredictBatchRequest(BaseModel):
    texts: List[constr(min_length=1, max_length=5000)]

@router.post("/predict")
def predict_single(req: PredictRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    cleaned_text = preprocess_text(req.text)
    if not cleaned_text:
        return {"label": "Netral", "score": 0.0} # Cannot predict empty text after cleaning

    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)[0]

    # Calculate confidence score if possible (using decision function for SVM)
    try:
        decision = model.decision_function(X)[0]
        # Normalize decision to a 0-1 range for a pseudo confidence score
        # For a linear SVM, we can use a sigmoid to map decision value to probability
        import math
        score = 1 / (1 + math.exp(-decision))
        if prediction.lower() == 'negatif':
            score = 1 - score # Ensure score represents confidence in the predicted class
    except Exception:
        score = 0.8 # Fallback

    return {
        "label": prediction.capitalize(),
        "score": round(max(score, 1 - score), 4) # Confidence in the chosen class
    }

@router.post("/predict-batch")
def predict_batch(req: PredictBatchRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100.")
        
    results = []
    for text in req.texts:
        cleaned_text = preprocess_text(text)
        if not cleaned_text:
            results.append({"label": "Netral", "score": 0.0})
            continue
            
        X = vectorizer.transform([cleaned_text])
        prediction = model.predict(X)[0]
        
        try:
            decision = model.decision_function(X)[0]
            import math
            score = 1 / (1 + math.exp(-decision))
            if prediction.lower() == 'negatif':
                score = 1 - score
        except Exception:
            score = 0.8
            
        results.append({
            "label": prediction.capitalize(),
            "score": round(max(score, 1 - score), 4)
        })
        
    return {"results": results}
