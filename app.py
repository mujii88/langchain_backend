from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import chromadb


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
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = "gemini-1.5-flash"

# Path to resume
path = '/home/one/Downloads/resume_NKkzjsrD_1756298749258.pdf'

# Load PDF
def load():
    loader = PyPDFLoader(path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

document = load()

# Split into chunks
text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
)

chunks = text_splitters.split_documents(document)

# Prepare docs for Chroma
docs = [Document(
    page_content=d.page_content,
    metadata={"source": "resume_NKkzjsrD_1756298749258.pdf"}
) for d in chunks]

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    api_key=GEMINI_API_KEY,
    model="models/text-embedding-004"
)

# Connect to Chroma Cloud
client = chromadb.CloudClient(
    api_key='ck-A35p4rqdu1odMeuZezLBJNJFv4GtnLjh7VLjfUFDcLAT',
    tenant='f61c9500-421c-4013-aff0-cc7813af5ff6',
    database='portfolio'
)

# Ensure fresh collection with correct embedding size
try:
    client.delete_collection("resume")
except Exception:
    pass  # ignore if collection doesn't exist

chroma = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="resume",
    client=client
)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query is required")
        
        query = request.query
        results = chroma.similarity_search(query, k=1)

        if not results:
            return SearchResponse(about_mujtaba="No relevant information found for your query.")

        # Format retrieved context
        temp = "\n".join([f"Result {i+1}: {doc.page_content}" for i, doc in enumerate(results)])

        # Run Gemini model
        llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, temperature=0.9, model=gemini_model)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            ("user", "{user}")
        ])

        system = 'Your task is to format the response about Mujtaba Ahmed from the given Query' 
        user = temp 

        chain = (
            {'system': lambda x: x['system'],
             'user': lambda x: x['user']}
            | prompt
            | llm
            | (lambda x: x.content)
        )

        response_content = chain.invoke({"system": system, "user": user})
        return SearchResponse(about_mujtaba=response_content)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Mujtaba Ahmed RAG API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
