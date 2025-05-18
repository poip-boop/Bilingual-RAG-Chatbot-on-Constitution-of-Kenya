Lawyer Chatbot
Overview
Lawyer Chatbot is a web application that provides answers to legal questions based on the Kenyan Constitution. It uses a Retrieval-Augmented Generation (RAG) approach to retrieve relevant sections of the Constitution and generate accurate responses. The application supports questions in English and Swahili, with automatic language detection and translation. Built with FastAPI, it serves a simple frontend (index.html) and exposes an API endpoint for querying.
Features

Legal Query Answering: Answers questions about the Kenyan Constitution using RAG.
Multilingual Support: Handles questions in English and Swahili with automatic translation.
FastAPI Backend: Provides a REST API for querying the chatbot.
ChromaDB for Vector Storage: Stores embeddings of the Constitution for efficient retrieval.
Groq API Integration: Generates responses using the Groq language model.

Prerequisites
Before running the project, ensure you have the following installed:

Python 3.8 or higher
Git
A Groq API key (sign up at Groq to obtain one)

Installation

Clone the Repository:
git clone https://github.com/poip-boop/Bilingual-RAG-Chatbot-on-Constitution-of-Kenya.git
cd lawyer-chatbot


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Ensure you have a requirements.txt file with the following dependencies (or run the command below to generate it):
fastapi
uvicorn
nest-asyncio
python-dotenv
langdetect
deep-translator
pdfplumber
spacy
sentence-transformers
chromadb
groq

Install them with:
pip install -r requirements.txt

Additionally, download the spaCy model:
python -m spacy download en_core_web_sm


Set Up Environment Variables:Create a .env file in the project root and add your Groq API key:
GROQ_API_KEY=your-groq-api-key

Note: Never commit the .env file to Git. Ensure it’s listed in .gitignore.

Prepare the Constitution PDF:Place the Kenyan Constitution PDF (COK.pdf) in the project root. The application will process this file to build the knowledge base.

Create a Frontend:Ensure an index.html file exists in the project root to serve as the frontend. A basic example:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lawyer Chatbot</title>
</head>
<body>
    <h1>Lawyer Chatbot</h1>
    <input type="text" id="question" placeholder="Ask a question about the Kenyan Constitution">
    <select id="language">
        <option value="en">English</option>
        <option value="sw">Swahili</option>
    </select>
    <button onclick="askQuestion()">Ask</button>
    <p id="answer"></p>
    <script>
        async function askQuestion() {
            const question = document.getElementById("question").value;
            const language = document.getElementById("language").value;
            const response = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, language })
            });
            const data = await response.json();
            document.getElementById("answer").innerText = data.answer;
        }
    </script>
</body>
</html>



Usage

Run the Application:Start the FastAPI server with:
python main.py

The application will:

Process the COK.pdf file and store embeddings in ChromaDB (first run only).
Start a server on http://localhost:8000 (or the port specified in your environment).


Access the Chatbot:

Open http://localhost:8000 in your browser to access the frontend.
Enter a question about the Kenyan Constitution (e.g., "What are the rights to privacy in Kenya?").
Select the language (English or Swahili) and click "Ask" to get a response.


API Usage:You can also query the chatbot directly via the API:
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What are the rights to privacy in Kenya?", "language": "en"}'



Project Structure

.history
_pycache_
chroma_db/: Directory where ChromaDB stores the embedded Constitution data.
.gitignore:Containing secret keys.
RAG_logic.py: Handles the RAG pipeline (PDF processing, embedding, retrieval, and response generation).
COK.pdf: The Kenyan Constitution PDF (not included in the repository; you must provide this).
Requrements.txt: Prerequisites to execute the program.
index.html: The frontend interface for interacting with the chatbot.
main.py: The FastAPI application serving the frontend and API endpoints.


Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

FastAPI for the web framework.
ChromaDB for vector storage.
Groq for the language model API.
Sentence Transformers for text embeddings.

