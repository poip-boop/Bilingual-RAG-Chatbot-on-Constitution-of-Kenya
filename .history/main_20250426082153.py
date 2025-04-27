from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import os
from langdetect import detect
from deep_translator import GoogleTranslator

from RAG_logic import setup_knowledge_base, query_constitution, generate_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    html = Path("index.html").read_text()
except FileNotFoundError:
    html = "<h1>Error: index.html not found</h1>"

@app.on_event("startup")
async def startup_event():
    try:
        setup_knowledge_base()
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=html)

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        target_language = data.get("language", None)

        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question.")

        input_language = detect(question)
        translated_question = (
            GoogleTranslator(source='auto', target='en').translate(question)
            if input_language != 'en' else question
        )

        relevant_chunks = query_constitution(translated_question)
        context = "\n".join(relevant_chunks)
        answer_en = generate_response(translated_question, context)

        response_lang = target_language or input_language
        answer = (
            GoogleTranslator(source='en', target=response_lang).translate(answer_en)
            if response_lang != 'en' else answer_en
        )

        return {"answer": answer, "language": response_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
