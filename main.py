import json
from fastapi import FastAPI
from pydantic import BaseModel
from nova.vector.ingest import ingest_documents
from nova.scripts.chat import chat_with_context

app = FastAPI()

# Load settings on boot
with open("settings.json", "r") as f:
    settings = json.load(f)

# Run the smart ingestion when the server starts up
@app.on_event("startup")
def startup_event():
    print(">> Checking documents for updates...")
    # Notice we add "/vector_store" here so it perfectly matches db.py!
    ingest_documents(settings["scan_root"], f"{settings['vector_db_path']}/vector_store")
    print(">> N.O.V.A. API is ready!")

# Define the structure of the data Flutter will send
class QueryRequest(BaseModel):
    question: str

# Create the endpoint Flutter will call
@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    answer = chat_with_context(
        question=request.question, 
        db_path=settings["vector_db_path"], 
        model_path=settings["model_path"]
    )
    return {"response": answer}

# Note: You run this using 'uvicorn main:app --host 0.0.0.0 --port 8000' in your Dockerfile