import os
import json
import pickle
import faiss
import numpy as np
from nova.vector.embed import embed_text

# Document parsing libraries
import pypdf
import docx
import pandas as pd

# Helper to track which files we've already processed
def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {}

def save_state(state, state_path):
    with open(state_path, "w") as f:
        json.dump(state, f)

def split_into_chunks(text, max_chars=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end].strip())
        start += max_chars - overlap
    return chunks

def extract_text(file_path, ext):
    """Extracts text from various file formats."""
    text = ""
    try:
        if ext == '.txt':
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == '.pdf':
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif ext == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            text = df.to_string()
        elif ext == '.xlsx':
            df = pd.read_excel(file_path, engine='openpyxl')
            text = df.to_string()
    except Exception as e:
        print(f"[WARN] Could not read {file_path}. Skipping. Error: {e}")
    return text

def ingest_documents(folder_path: str, save_path: str = "db/vector_store"):
    index_path = f"{save_path}.index"
    pkl_path = f"{save_path}.pkl"
    state_path = f"{save_path}_state.json"
    
    state = load_state(state_path)
    new_texts = []
    new_metadata = []  # NEW: Track metadata alongside texts
    updated_state = state.copy()

    supported_exts = ['.txt', '.pdf', '.docx', '.csv', '.xlsx']

    ignore_dirs = {
        'build', '.git', 'node_modules', '__pycache__', 'intermediates', 
        'linux', 'windows', 'macos', 'ios', 'android', 'runner', 
        'gerber', 'production', 'export', 'Symbols_Footprints_and_Models' 
    }
    
    skip_keywords = {'raw data', 'test phase', 'dataset', 'labels', 'cmake'}

    for root, dirs, files in os.walk(folder_path):
        if ".nova_ignore" in files:
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if d not in ignore_dirs 
                   and not d.startswith('.') 
                   and d.lower() not in skip_keywords]

        for filename in files:
            if filename.lower() in ['cmakelists.txt', 'license.txt']:
                continue
            
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in supported_exts:
                file_path = os.path.join(root, filename)
                
                if ext == '.txt' and os.path.getsize(file_path) < 50:
                    continue

                rel_path = os.path.relpath(file_path, folder_path)
                
                try:
                    mtime = os.path.getmtime(file_path)
                except FileNotFoundError:
                    continue
                
                if rel_path not in state or state[rel_path] < mtime:
                    print(f"[INFO] Extracting: {rel_path}")
                    
                    full_text = extract_text(file_path, ext)
                    
                    if full_text.strip():
                        chunks = split_into_chunks(full_text)
                        
                        # NEW: Capture folder and add to parallel lists
                        for chunk in chunks:
                            new_texts.append(chunk)
                            folder = os.path.dirname(rel_path)
                            if not folder: 
                                folder = "root"
                                
                            new_metadata.append({
                                "folder": folder, 
                                "filename": os.path.basename(rel_path)
                            })
                        
                    updated_state[rel_path] = mtime

    if not new_texts:
        print("[INFO] No new documents to ingest. Database is up to date.")
        return

    print(f"[INFO] Ingesting {len(new_texts)} new chunks...")
    new_embeddings = embed_text(new_texts)
    new_embeddings = np.array(new_embeddings).astype("float32")

    if os.path.exists(index_path) and os.path.exists(pkl_path):
        index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            all_texts = pickle.load(f)
    else:
        dim = new_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        all_texts = []

    index.add(new_embeddings)
    
    # NEW: Save as dictionaries containing the text AND metadata
    combined_chunks = [{"text": t, "folder": m["folder"], "filename": m["filename"]} for t, m in zip(new_texts, new_metadata)]
    all_texts.extend(combined_chunks)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(all_texts, f)
    save_state(updated_state, state_path)
    
    print("[INFO] Vector database updated successfully.")