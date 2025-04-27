from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import os

from RAg_logic import setup_knowledge_base, query_constitution, generate_response

app = FastAPI()

# Add CORS middleware to allow frontend requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production (e.g., ["https://your-frontend.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load index.html (ensure it's in the project root)
try:
    html = Path("index.html").read_text()
except FileNotFoundError:
    html = "<h1>Error: index.html not found</h1>"

@app.on_event("startup")
async def startup_event():
    try:
        setup_knowledge_base()  # Load and embed Constitution
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
        language = data.get("language", "en")

        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question.")

        relevant_chunks = query_constitution(question)
        context = "\n".join(relevant_chunks)
        answer = generate_response(question, context)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    # Use environment variable for port, default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)