@echo off
echo Starting FastAPI Backend for DeepSentiment...
cd backend
python -m uvicorn app:app --reload --port 8000
