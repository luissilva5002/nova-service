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
    updated_state = state.copy()

    supported_exts = ['.txt', '.pdf', '.docx', '.csv', '.xlsx']

    # Add these to your ingest_documents function
    ignore_dirs = {
        'build', '.git', 'node_modules', '__pycache__', 'intermediates', 
        'linux', 'windows', 'macos', 'ios', 'android', 'runner', # App build junk
        'gerber', 'production', 'export', 'Symbols_Footprints_and_Models' # PCB junk
    }
    
    # Specific folder names to skip entirely
    skip_keywords = {'raw data', 'test phase', 'dataset', 'labels', 'cmake'}

    for root, dirs, files in os.walk(folder_path):
        # 1. Skip if .nova_ignore is present
        if ".nova_ignore" in files:
            dirs[:] = []
            continue

        # 2. Strict Folder Filtering
        dirs[:] = [d for d in dirs if d not in ignore_dirs 
                   and not d.startswith('.') 
                   and d.lower() not in skip_keywords]

        # 3. File Filtering
        for filename in files:
            # Skip massive data files or system files
            if filename.lower() in ['cmakelists.txt', 'license.txt']:
                continue

        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in supported_exts:
                file_path = os.path.join(root, filename)
                
                # OPTIONAL: Skip files that are too small or look like ML labels
                # (e.g., skip .txt files under 50 bytes because they are likely just IDs)
                if ext == '.txt' and os.path.getsize(file_path) < 50:
                    continue

                rel_path = os.path.relpath(file_path, folder_path)
                
                try:
                    mtime = os.path.getmtime(file_path)
                except FileNotFoundError:
                    continue # Skip broken files or symlinks
                
                # If it's a new file, or it has been modified since we last checked
                if rel_path not in state or state[rel_path] < mtime:
                    print(f"[INFO] Extracting: {rel_path}")
                    
                    full_text = extract_text(file_path, ext)
                    
                    if full_text.strip():
                        chunks = split_into_chunks(full_text)
                        new_texts.extend(chunks)
                        
                    updated_state[rel_path] = mtime

    if not new_texts:
        print("[INFO] No new documents to ingest. Database is up to date.")
        return

    # 2. Embed the new text
    print(f"[INFO] Ingesting {len(new_texts)} new chunks...")
    new_embeddings = embed_text(new_texts)
    new_embeddings = np.array(new_embeddings).astype("float32")

    # 3. Load existing DB (or create new) and append
    if os.path.exists(index_path) and os.path.exists(pkl_path):
        index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            all_texts = pickle.load(f)
    else:
        dim = new_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        all_texts = []

    index.add(new_embeddings)
    all_texts.extend(new_texts)

    # 4. Save everything
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(all_texts, f)
    save_state(updated_state, state_path)
    
    print("[INFO] Vector database updated successfully.")