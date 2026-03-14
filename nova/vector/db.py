import faiss
import pickle
import os
import numpy as np

def _paths(db_path: str):
    # Handle variations in how the path might be passed
    if db_path.endswith("vector_store"):
        return db_path + ".index", db_path + ".pkl"
    return os.path.join(db_path, "vector_store.index"), os.path.join(db_path, "vector_store.pkl")

def get_available_folders(db_path: str) -> list[str]:
    """Reads the metadata and returns a list of all available folders."""
    index_file, pkl_file = _paths(db_path)
    if not os.path.exists(pkl_file):
        return []
        
    with open(pkl_file, "rb") as f:
        texts = pickle.load(f)
        
    folders = set()
    for item in texts:
        if isinstance(item, dict) and "folder" in item:
            folders.add(item["folder"])
            
    return list(folders)

def search_db(query_vector, db_path, return_scores=False, top_k=4, target_folder="ALL"):
    index_file, pkl_file = _paths(db_path)
    
    index = faiss.read_index(index_file)
    with open(pkl_file, "rb") as f:
        texts = pickle.load(f)

    # Fetch a larger batch if filtering, to ensure we get enough chunks to fill top_k
    fetch_k = top_k * 15 if target_folder != "ALL" else top_k
    fetch_k = min(fetch_k, len(texts)) 

    distances, indices = index.search(np.array([query_vector]).astype("float32"), fetch_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(texts):
            item = texts[idx]
            
            # Extract text and folder (supports new dict format and falls back safely)
            if isinstance(item, dict):
                text = item.get("text", "")
                folder = item.get("folder", "root")
            else:
                text = item
                folder = "root"

            # Skip chunk if it doesn't match the LLM's chosen folder
            if target_folder != "ALL" and folder != target_folder:
                continue

            score = distances[0][i]
            if return_scores:
                results.append((text, score))
            else:
                results.append(text)
                
            if len(results) >= top_k:
                break

    return results