from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# Import your modules
# Ensure rag_gemini is importable. If you run this from the root folder, this works.
from rag_chatbot.rag_gemini import rag_qa as gemini_qa
from rag_chatbot.rag_llama import rag_llama_qa as llama_qa
from rag_chatbot.rules import check_rules

app = FastAPI(title="Academic RAG Backend")

# ============================
# CORS (Important for Frontend)
# ============================
# This allows your friend's frontend (running on a different port) to talk to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Data Models
# ============================
class ChatRequest(BaseModel):
    question: str
    model_type: str = "hybrid"  # Options: "gemini", "llama", "hybrid"

class ChatResponse(BaseModel):
    answer: str
    source: str

# ============================
# Routes
# ============================
@app.get("/")
def read_root():
    return {"status": "running", "message": "Academic RAG Backend is ready."}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    question = request.question
    model = request.model_type.lower()
    
    answer = ""
    source_used = ""

    # 1. HYBRID LOGIC (Optional: Check rules first for all models or just hybrid)
    # If the user selected 'hybrid', we check hardcoded rules first.
    if model == "hybrid":
        rule_answer = check_rules(question)
        if rule_answer:
            return ChatResponse(answer=rule_answer, source="Rule-Based")
        
        # If no rule matches, fallback to Gemini (or you can make it fallback to Llama)
        try:
            answer = gemini_qa(question)
            source_used = "Gemini (Hybrid Fallback)"
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 2. GEMINI ONLY
    elif model == "gemini":
        try:
            answer = gemini_qa(question)
            source_used = "Gemini Cloud"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

    # 3. LOCAL LLAMA ONLY
    elif model == "llama":
        try:
            answer = llama_qa(question)
            source_used = "Local Llama"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Llama Error: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'gemini', 'llama', or 'hybrid'.")

    return ChatResponse(answer=answer, source=source_used)

# Run with: uvicorn main_backend:app --reload