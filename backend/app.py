from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import predict, dashboard

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for the static frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(dashboard.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
