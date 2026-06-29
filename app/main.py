from fastapi import FastAPI
from app.routes import llm

app = FastAPI()

@app.get("/")
def landing_page():
    return {
        "message": "Hi this your Backend server"
    }

@app.get("/health")
def health():
    return {
        "status": " ok"
    }

app.include_router(llm.router)


