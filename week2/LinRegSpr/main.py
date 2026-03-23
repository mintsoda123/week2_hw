import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(title="Hooke's Law ML Demo")
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/train")
async def train():
    try:
        from train_model import train_and_evaluate
        result = train_and_evaluate()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


class PredictReq(BaseModel):
    mass: float


@app.post("/api/predict")
async def predict(req: PredictReq):
    try:
        from train_model import predict_length
        length = predict_length(req.mass)
        theoretical = 2.0 * req.mass + 10.0
        return JSONResponse(content={
            "mass": req.mass,
            "predicted": round(length, 4),
            "theoretical": round(theoretical, 4),
            "error_pct": round(abs(length - theoretical) / theoretical * 100, 4),
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
