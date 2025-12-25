from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from pydantic import BaseModel
from .rag import answer_question
from .data_store import load_document, embed_chunks


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

