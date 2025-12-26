from dotenv import load_dotenv
load_dotenv()

import os


from fastapi import FastAPI , UploadFile, File
from pydantic import BaseModel
from .rag import answer_question
from .data_store import load_document, embed_chunks, reload_document


app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("🚀 Backend starting up...")
    load_document("data/AI_gentle_intro.pdf")
    print("📄 Document loaded and chunked")
    embed_chunks()
    print("📄 Document embedded and ready")

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)
    return {
        "question": request.question,
        "answer": answer
    }

UPLOAD_DIR = "uploads"

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    reload_document(file_path)

    return {
        "filename": file.filename,
        "status": "uploaded and activated"
    }



