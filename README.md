N.O.V.A. (Neural Optimizer and Virtual Assistant)

N.O.V.A. is a fully local, privacy-first AI assistant. Built entirely from the ground up, it features a custom LoRA-tuned LLM running via llama.cpp and an advanced Hierarchical Retrieval-Augmented Generation (RAG) system powered by FAISS.

Unlike standard AI wrappers, N.O.V.A. doesn't just read documents blindly—it understands your folder structure, intelligently routing questions to specific domains (like "Hardware Components" vs "Python Courses") to completely eliminate cross-domain hallucinations.

🏗️ System Architecture

N.O.V.A. is containerized via Docker and served through a FastAPI backend. The core logic is split into two main domains: Vector Memory and LLM Orchestration.

Plaintext

C:.

├── main.py              # FastAPI entry point & API routes

├── settings.json        # Global configuration (models, chunk size, thresholds)

├── docker-compose.yml   # Maps local folders to the container for fast iteration

└── nova/

├── scripts/

│   └── chat.py      # Brain of N.O.V.A. (Query routing, prompts, llama.cpp generation)

└── vector/

├── db.py        # FAISS search engine and metadata filtering

├── embed.py     # SentenceTransformers text-to-vector encoding

└── ingest.py    # Document parsing, chunking, and metadata extraction

🧠 How the Memory Works (Hierarchical RAG)

Standard RAG pipelines dump all text into a single, flat vector database. If you ask a question about Python, the math might accidentally pull a paragraph from a legally binding patent just because it shares keywords.

N.O.V.A. solves this using a Metadata-Filtered Hierarchical RAG approach. Here is how it works under the hood:

1. Ingestion & Embedding (ingest.py & embed.py)

When N.O.V.A. boots up, it scans the scan\_root folder for new or modified documents (.txt, .pdf, .csv, etc.).

Extraction & Chunking: It parses the files and splits the text into chunks (e.g., 500 characters) so the LLM can digest them easily.

Metadata Tagging: As it chunks the text, it records the exact folder the file came from (e.g., EITCA ML COURSE/3 - Functions).

Mathematical Embedding: It passes the text chunk to all-MiniLM-L6-v2 (via embed.py), which translates the English text into a dense array of floating-point numbers (a Vector).

Dual-Storage: \* The vector is stored in a FAISS Index (vector\_store.index) for hyper-fast math calculations.

The actual text and its folder metadata are saved in a parallel Pickle File (vector\_store.pkl) as a dictionary: {"text": "...", "folder": "EITCA ML COURSE"}.

1. The Smart Query Router (chat.py)

When a user asks a question, N.O.V.A. does not search the database immediately.

db.py reads the .pkl file and rolls up all deep folder paths into top-level categories (max 2 levels deep) to save tokens.

The LLM is fed a tiny, lightning-fast prompt containing the user's question and a list of available top-level folders.

The LLM acts as a Router, deciding which specific folder to search. If the question is about hardware, it outputs NG/COMPONENTS. If it's a general coding question, it outputs ALL.

1. Filtered Vector Search (db.py)

Once the router picks a target folder, the actual vector search begins:

The user's question is embedded into a vector.

FAISS performs an L2 Distance Search across the database, finding the text chunks that are mathematically closest to the question's meaning. We fetch a large batch of results.

The Metadata Filter: db.py loops through the FAISS results, instantly discarding any chunks that do not belong to the Router's chosen folder.

It returns the Top K closest matches that survived the filter.

1. Answer Generation (chat.py)

Finally, N.O.V.A. checks the similarity scores of the retrieved chunks.

If the chunks are highly relevant (below the distance threshold), they are injected into the context of the System Prompt, and N.O.V.A. formulates a factual answer.

If no relevant chunks are found, N.O.V.A. defaults to its general knowledge/coding capabilities, refusing to hallucinate document facts.

🚀 Running N.O.V.A.

N.O.V.A. is designed to be run locally without relying on external cloud APIs.

Prerequisites

Docker & Docker Compose

Local GGUF Model (e.g., nova\_model\_lora.gguf) placed in the models/ folder.

Booting the System

Configure your local paths in docker-compose.yml to mount your documents, DB, and models.

YAML

volumes:

- C:\Path\To\Your\Docs:/app/data/scan\_root
- ./db:/app/db
- ./models:/app/models

Build and start the container:

Bash

docker-compose up --build

Watch the logs. N.O.V.A. will automatically ingest any new documents in the mounted folder on startup.

Interacting via API

You can talk to N.O.V.A. by sending a POST request to the FastAPI backend:

Bash

curl -X POST "http://localhost:8000/chat" \

- H "Content-Type: application/json" \
- d '{"question": "Can you check the Hardware folder and tell me the battery specs?"}'

🛠️ Configuration (settings.json)

You can tweak N.O.V.A.'s brain without changing code:

context\_window: Set to 2048 (or 4096 for massive context reading).

max\_chunk\_size: How large document paragraphs should be when saved to memory.

top\_k\_results: How many document chunks to feed the LLM per query.
