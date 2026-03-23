"""
FastAPI backend — Hooke's Law ML Predictor with Min-Max Normalization
Run:  uvicorn main:app --reload --port 8000
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path

from model.hookes_model import get_model

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hooke's Law ML Predictor",
    description="TensorFlow + Min-Max Normalization Demo",
    version="1.0.0",
)

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
TEMPLATES   = Jinja2Templates(directory=str(BASE_DIR / "templates"))

OUTPUT_DIR.mkdir(exist_ok=True)


# ── Schemas ───────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    epochs: int         = Field(default=1500, ge=100, le=5000,
                                description="Number of training epochs")
    learning_rate: float = Field(default=0.001, ge=1e-5, le=0.1,
                                 description="Adam optimizer learning rate")


class PredictRequest(BaseModel):
    mass_kg: float = Field(ge=0.01, le=200.0,
                           description="Mass in kilograms (0.01 ~ 200 kg)")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main SPA."""
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.post("/api/train")
async def train(req: TrainRequest):
    """
    Train the TensorFlow model on Hooke's Law data.
    Applies Min-Max Normalization before training (same as 03_data_preprocessing.py).
    Generates 3 PNG plots: normalization, loss curve, regression fit.
    """
    try:
        model   = get_model()
        metrics = model.train(epochs=req.epochs, learning_rate=req.learning_rate)
        plots   = [f.name for f in OUTPUT_DIR.glob("*.png")]
        return {
            "status":  "success",
            "metrics": metrics,
            "plots":   sorted(plots),
            "message": (
                f"Training complete — R² = {metrics['r2_score']:.4f} "
                f"({metrics['accuracy_pct']:.2f}%)"
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    Predict spring displacement for a given mass.
    Requires the model to be trained first.
    Generates 04_prediction_result.png.
    """
    model = get_model()
    if not model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. POST to /api/train first.",
        )
    try:
        result = model.predict(req.mass_kg)
        result["plot"] = "04_prediction_result.png"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/status")
async def status():
    """Return current model status and available plots."""
    model = get_model()
    plots = sorted(f.name for f in OUTPUT_DIR.glob("*.png"))
    return {
        "is_trained":  model.is_trained,
        "metrics":     model.metrics,
        "plots":       plots,
        "scaler_x":    model.scaler_x.to_dict() if model.is_trained else {},
        "scaler_y":    model.scaler_y.to_dict() if model.is_trained else {},
    }


@app.get("/output/{filename}")
async def get_output(filename: str):
    """Serve a PNG plot from the output directory."""
    safe = Path(filename).name          # prevent path traversal
    path = OUTPUT_DIR / safe
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(str(path), media_type="image/png",
                        headers={"Cache-Control": "no-cache"})


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
