import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.rag import UploadResponse, ChatRequest, ChatResponse
from app.services.rag_service import rag_service

router = APIRouter()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")
    
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks_count = rag_service.process_pdf(file_path)
        return UploadResponse(
            filename=file.filename,
            chunks=chunks_count,
            message="Document indexé avec succès."
        )
    except Exception as e:
        print(f"❌ Erreur Upload : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/chat", response_model=ChatResponse)
def chat_with_pdf(request: ChatRequest):
    try:
        res = rag_service.answer_question(request.question)
        return ChatResponse(
            answer=res["answer"],
            sources=res["sources"]
        )
    except Exception as e:
        print(f"❌ Erreur Chat : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))