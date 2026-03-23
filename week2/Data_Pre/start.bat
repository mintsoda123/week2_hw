@echo off
cd /d "%~dp0"
echo Starting Hooke's Law ML Predictor...
echo Open browser at: http://localhost:8002
..\tf_env\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
