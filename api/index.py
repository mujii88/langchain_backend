from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb

# Initialize FastAPI
app = FastAPI(
    title="Mujtaba Ahmed RAG API",
    description="RAG system for querying information about Mujtaba Ahmed"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    about_mujtaba: str

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = "gemini-1.5-flash"

# Initialize components
def initialize_components():
    # Load PDF (you'll need to upload this file to Vercel or use a URL)
    # For now, we'll initialize with empty documents
    documents = []
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        api_key=GEMINI_API_KEY,
        model="models/text-embedding-004"
    )
    
    # Initialize Chroma
    try:
        client = chromadb.HttpClient(
            host='your-chroma-host',  # Replace with your Chroma host
            port=8000,
            ssl=False
        )
        
        # Create or get collection
        collection = client.get_or_create_collection("resume")
        
        # Initialize Chroma with the collection
        chroma = Chroma(
            client=client,
            collection_name="resume",
            embedding_function=embeddings
        )
        
        return chroma, embeddings
    except Exception as e:
        print(f"Error initializing Chroma: {str(e)}")
        # Fallback to in-memory Chroma for testing
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="resume"
        ), embeddings

# Initialize components when the app starts
chroma, embeddings = initialize_components()

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        # Your search logic here
        # Example:
        results = chroma.similarity_search(request.query, k=3)
        context = "\n".join([doc.page_content for doc in results])
        
        return SearchResponse(
            about_mujtaba=context[:500]  # Limit response length
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Mujtaba Ahmed RAG API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
