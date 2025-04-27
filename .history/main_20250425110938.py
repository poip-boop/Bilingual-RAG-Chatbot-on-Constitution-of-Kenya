# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
import uvicorn
import nest_asyncio

from rag_logic import setup_knowledge_base, query_constitution, generate_response

nest_asyncio.apply()
app = FastAPI()

html = Path("index.html").read_text()

@app.on_event("startup")
def startup_event():
    setup_knowledge_base()  # Load and embed Constitution

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=html)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    language = data.get("language", "en")

    if not question:
        return {"answer": "Please provide a question."}

    relevant_chunks = query_constitution(question)
    context = "\n".join(relevant_chunks)
    answer = generate_response(question, context)

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
