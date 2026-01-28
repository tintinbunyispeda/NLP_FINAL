import os
from dotenv import load_dotenv
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from google import genai

# =========================
# ENV & GEMINI
# =========================
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "models/gemini-2.5-flash"

# =========================
# LOAD DOCUMENTS
# =========================
def load_documents():
    docs = []
    data_dir = "data"

    for file in os.listdir(data_dir):
        if file.endswith(".md"):
            path = os.path.join(data_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": file}
                    )
                )
    return docs

# =========================
# VECTOR STORE
# =========================
def build_vectorstore():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

print("🔄 Building vector store...")
vectorstore = build_vectorstore()
print("✅ Vector store ready.")

# =========================
# RAG FUNCTION (FIXED)
# =========================
def rag_qa(question: str) -> str:
    retrieved_docs = vectorstore.similarity_search(question, k=3)

    if not retrieved_docs:
        return "I don't know based on the provided documents."

    context = "\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = f"""
You are an academic assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text

# =========================
# CLI CHAT LOOP
# =========================
#if __name__ == "__main__":
    print("\n📚 RAG Chatbot (type 'exit' to quit)\n")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        answer = rag_qa(q)
        print("\nBot:", answer, "\n")

# alias biar compatible sama main.py
def tanya(pertanyaan: str) -> str:
    return rag_qa(pertanyaan)
