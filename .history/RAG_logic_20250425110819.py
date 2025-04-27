# %%
# Change to python,swahili ,no need to rerun everytime
# Importing libraries
import os
import warnings
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from groq import Groq
import uuid


# %%
# supress warnings
warnings.filterwarnings("ignore")


# %%
# Load environment variables from .env file
load_dotenv()

# %%
# Retrieve API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Use the string "GROQ_API_KEY"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# %%
# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# %%
# Initialize spaCy for text processing
nlp = spacy.load("en_core_web_sm")

# %%
# Initialize sentence transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# %%
# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="constitution")

# %%
# Path to the Kenyan Constitution PDF 
PDF_PATH = r"C:\Users\Administrator\Documents\PROJECTS\Lawyer chatbot\COK.pdf"

# %%
# Extract text
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

# %%
# Chunk Text
def chunk_text(text, max_tokens=500):
    """Chunk text into smaller pieces for processing."""
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sent in doc.sents:
        sent_text = sent.text
        sent_tokens = len(nlp(sent_text))
        if current_length + sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent_text
            current_length = sent_tokens
        else:
            current_chunk += " " + sent_text
            current_length += sent_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# %%
# Embed and store chunks
def embed_and_store(chunks):
    """Embed text chunks and store in ChromaDB."""
    embeddings = embedder.encode(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[str(uuid.uuid4())]
        )

# %%
# Query function
def query_constitution(query, n_results=5):
    """Query the constitution database and get relevant results."""
    # Embed the query
    query_embedding = embedder.encode([query])[0]
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    return results['documents'][0]

# %%
# Response function
def generate_response(query, context):
    """Generate a response using Groq API."""
    prompt = f"""
    You are a legal assistant specializing in the Kenyan Constitution. Based on the following context from the Kenyan Constitution, answer the query accurately and concisely. If the context is insufficient, indicate so and provide a general response based on your knowledge.

    Context:
    {context}

    Query:
    {query}

    Answer:
    """
    
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a knowledgeable legal assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=500
    )
    
    return response.choices[0].message.content


# %%
def main():
    # Check if the database is already populated
    if collection.count() == 0:
        print("Processing Kenyan Constitution PDF...")
        # Extract text from PDF
        text = extract_text_from_pdf(PDF_PATH)
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Embed and store chunks
        embed_and_store(chunks)
        print("Constitution data processed and stored in ChromaDB.")
    else:
        print("Using existing ChromaDB collection.")

    # Example query loop
    while True:
        query = input("\nEnter your question about the Kenyan Constitution (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        # Query the database
        relevant_chunks = query_constitution(query)
        context = "\n".join(relevant_chunks)
        
        # Generate response
        response = generate_response(query, context)
        print("\nAnswer:", response)

if __name__ == "__main__":
    main()


