from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import nest_asyncio
import uvicorn
from pathlib import Path

# Allow running in notebooks (safe even if not used there)
nest_asyncio.apply()

app = FastAPI()

# Load HTML template
html_path = Path(r"C:\Users\Administrator\Documents\PROJECTS\Lawyer chatbot\index.html")
html = html_path.read_text()

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=html)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    language = data.get("language", "en")

    if not question:
        return {"answer": "Please provide a question"}

    try:
        # Replace with your actual RAG logic here
        

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
