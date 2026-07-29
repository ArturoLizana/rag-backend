import os
# Brider la mémoire et le multi-threading de PyTorch au démarrage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.rag_service import rag_service

app = FastAPI(title="Arturo RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "API RAG fonctionnelle"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Fichier non PDF")
    
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        num_chunks = rag_service.process_pdf(temp_path)
        return {"filename": file.filename, "message": "Indexé avec succès", "chunks": num_chunks}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat")
def chat(request: QueryRequest):
    result = rag_service.answer_question(request.question)
    # On renvoie la réponse ET les sources au frontend !
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }