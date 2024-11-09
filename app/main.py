
from typing import Dict, List
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  

from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime


from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

import sqlite3
import fitz  # PyMuPDF
import hashlib
import asyncio
import os
import json
import shutil
import uvicorn

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Define CORS origins
origins = ["http://localhost:3000", "http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup database connection
conn = sqlite3.connect('database.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    content TEXT,
    file_hash TEXT UNIQUE,
    uploaded_date TEXT
)
''')
conn.commit()

# Initialize LLM and embedding model
api_key = os.getenv("GROQ_API_KEY")
llm = Groq(model="llama3-8b-8192", api_key=api_key)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

# Directory for storing indexes
VECTOR_STORE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Path to the 'uploads' folder
UPLOADS_FOLDER = Path("uploads")

DATABASE_PATH = Path("backend/database.db")

# used for AI streaming response purpose
async def response_generator(response_stream):
    for text in response_stream:
        # Yield each chunk as a JSON string with a newline
        yield json.dumps({"chunk": str(text)}) + "\n"

# Helper function to fetch books from the database
def fetch_all_books() -> List[Dict]:
    try:
        # Connect to the database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Execute the query to fetch all books
        cursor.execute("SELECT * FROM documents")
        rows = cursor.fetchall()
        
        # Close the connection
        conn.close()
        
        # Format results as a list of dictionaries
        books = [row[1] for row in rows]
        return books
    
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
 
# Request model for questions
class QueryRequest(BaseModel):
    question: str
    document_id: str

# Define a model for a book (Optional, for better structure and type hints)
class Book(BaseModel):
    id: int
    filename: str

# Endpoint to check if API key has been provided
@app.get("/check-api-key/")
async def check_api_key():
    api_key = os.getenv("GROQ_API_KEY")  # Get the API key from environment variables
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is missing or invalid.")
    return {"message": "API key is present."}

# Upload PDF endpoint
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):

    # Check if the uploaded file is a PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

     # Read file content
    file_content = await file.read()
    file_path = f"uploads/{file.filename}"

     # Create uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Compute hash of the file content to check for duplicates
    file_hash = hashlib.sha256(file_content).hexdigest()


    # Check if this hash already exists in the database
    cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
    duplicate = cursor.fetchone()
    if duplicate:
        return {"message": "PDF has already been uploaded.", "duplicate": True}   

    # Extract text from the PDF
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])

    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Store text and metadata in SQLite
    cursor.execute("INSERT INTO documents (filename, content, file_hash, uploaded_date) VALUES (?, ?, ?, ?)", 
                   (file.filename, text, file_hash, current_date))
    
    document_id = cursor.lastrowid  # Retrieve the generated document_id
    conn.commit()

    # Create index ONCE during upload for efficient QnAs
    document = Document(text=text)
    index = VectorStoreIndex.from_documents([document])

    # Save index for later use
    vector_store_path = f"vector_stores/{document_id}"
    index.storage_context.persist(persist_dir=vector_store_path)
    
    return {"status": "success", "document_id": document_id, "filename": file.filename}

# Ask question endpoint
@app.post("/ask-question/")
def ask_question(request: QueryRequest):
    try:
        # LOAD the pre-computed index (fast)
        vector_store_path = f"vector_stores/{request.document_id}"
        storage_context = StorageContext.from_defaults(persist_dir=vector_store_path)
        index = load_index_from_storage(storage_context)
        
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

        chat_engine = index.as_chat_engine(
            stream=True,
            chat_mode="context",
            memory=memory,
            llm=llm,
            context_prompt=(
                "You are an expert chatbot, able to have normal interactions, as well as talk about any uploaded pdf file. You can also articulate concepts clearly"
                "Here are the relevant documents for the context:\n"
                "{context_str}"
                "\nInstruction: Use the previous chat history, or the context above, to interact and help the user. Please output a Complex Markdown of the response so that it can be presented well."
            ),
            verbose=True,
        )
        
       # Wrap the synchronous response in an async generator
        response_stream = chat_engine.stream_chat(request.question)
        
        async def response_generator():
            # Iterate through each text chunk in response_stream.response_gen
            for text in response_stream.response_gen:
                yield text  # Yield each text chunk to the client
                await asyncio.sleep(0)  # Yield control to the event loop

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


''' Additional endpoints for debugging purpose and other uses '''
# Endpoint for getting the recent uploads
@app.get("/list-uploads/")
async def list_uploads():
    # Check if the folder exists
    if not UPLOADS_FOLDER.exists() or not UPLOADS_FOLDER.is_dir():
        raise HTTPException(status_code=404, detail="Uploads folder not found")

    # List all PDF files in the folder
    pdf_files = fetch_all_books()

    # Check if any files were found
    if not pdf_files:
        return JSONResponse(content=[], status_code=200)

    return JSONResponse(content=pdf_files, status_code=200)

# Endpoint for fetching document_id needed for the LLM to fetchthe right vector embeddings
@app.get("/get-document-id/")
async def get_document_id(filename: str):

    # Construct the file path
    file_path = os.path.join(UPLOADS_FOLDER, filename)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Read file content and calculate the hash
    with open(file_path, "rb") as f:
        file_content = f.read()
        file_hash = hashlib.sha256(file_content).hexdigest()

    # Connect to the database and query by hash
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return {"document_id": result[0]}
    else:
        raise HTTPException(status_code=404, detail="Document not found")

# Endpoint to get all books
@app.get("/books/")
async def get_books():
    books = fetch_all_books()
    if not books:
        raise HTTPException(status_code=404, detail="No books found.")
    return books

# Delete PDF and index endpoint
@app.delete("/delete-pdf/{document_id}")
async def delete_pdf(document_id: int):
    try:
        # Fetch the document information from the database
        cursor.execute("SELECT filename FROM documents WHERE id = ?", (document_id,))
        document = cursor.fetchone()
        
        # Check if the document exists
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create a filepath 
        filename = document[0]
        file_path = f"uploads/{filename}"
        
        # Delete the PDF file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            raise HTTPException(status_code=404, detail="PDF file not found in uploads directory")

        # Remove document entry from the database
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
        
        # Delete the associated index directory in vector_stores
        vector_store_path = f"vector_stores/{document_id}"
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)

        # Fetch available books
        books_avail = fetch_all_books()

        # Check if books does not exist
        if not books_avail:
           
            # Instead of deleting the database, clear all entries in the documents table
            cursor.execute("DELETE FROM documents")
           
            conn.commit()
            
            # Delete vector_stores folder
            if os.path.exists(VECTOR_STORE_DIR):
                shutil.rmtree(VECTOR_STORE_DIR)
                print(f"{VECTOR_STORE_DIR} folder deleted.")
                
            # Delete uploads folder
            if os.path.exists(UPLOADS_FOLDER):
                shutil.rmtree(UPLOADS_FOLDER)
                print(f"{UPLOADS_FOLDER} folder deleted.")
        
        return {"status": "success", "message": f"{filename.strip('.pdf')} and its associated index deleted successfully"}
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete PDF and index")

# Get all documents endpoint
@app.get("/get-all-documents")
async def get_all_documents():
    try:
        cursor.execute("SELECT * FROM documents")
        documents = cursor.fetchall()
        
        # Check if any documents were retrieved
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")

        # Format the documents as a list of dictionaries with column names
        columns = [column[0] for column in cursor.description if column != "content"]  # Get column names
        documents_list = [dict(zip(columns, row)) for row in documents]

        return {"status": "success", "documents": documents_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.get('/')
def welcome():
    return "Welcome to PDF Chatter!"