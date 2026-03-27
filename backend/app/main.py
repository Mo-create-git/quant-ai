from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, upload, predict, results
from app import database

app = FastAPI(title="Quant-AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.include_router(results.router, prefix="/api/results", tags=["results"])

@app.get("/")
def root():
    return FileResponse("static/quant-ai.html")

@app.get("/upload")
def upload_page():
    return FileResponse("static/upload.html")

@app.get("/auth")
def auth_page():
    return FileResponse("static/auth.html")

@app.get("/results")
def results_page():
    return FileResponse("static/results.html")

# Mount static LAST
app.mount("/static", StaticFiles(directory="static"), name="static")
