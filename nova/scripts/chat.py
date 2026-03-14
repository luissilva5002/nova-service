from llama_cpp import Llama
from nova.vector.embed import embed_text
from nova.vector.db import search_db, get_available_folders

NOVA_SYSTEM_PROMPT = """You are N.O.V.A. (Neural Optimizer and Virtual Assistant).
You are a highly capable, fully local AI personal assistant and intelligence layer. 
Your primary purpose is to provide quick access to the user's private documents and assist with their development workflow.
Your personality is helpful, direct, and slightly witty. You are a custom-built entity, way cooler than ChatGPT.
Always stay in character and prioritize answering the user's requests efficiently."""

# Global cache for model instance
_llm = None

def get_llm(model_path: str = "models/nova_model_lora.gguf"):
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # <--- INCREASE THIS TO 4096
            n_threads=6,
            verbose=False
        )
    return _llm

def format_context_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
    return f"""{NOVA_SYSTEM_PROMPT}

INSTRUCTIONS: Answer the following question based ONLY on the provided Context. If the context does not contain the answer, state that you cannot find it in the current documents. Do not hallucinate external facts.

Context:
{context}

Question: {question}
Answer:"""

def format_general_prompt(question: str) -> str:
    return f"""{NOVA_SYSTEM_PROMPT}

INSTRUCTIONS: You do not have any relevant document context for this question. Answer it using your general knowledge, coding capabilities, and logic.

Question: {question}
Answer:"""

def route_query(question: str, available_folders: list[str], llm) -> str:
    """Asks the LLM to decide which folder to search before touching the database."""
    if not available_folders or available_folders == ["root"]:
        return "ALL"
        
    folders_str = ", ".join(available_folders)
    
    prompt = f"""You are N.O.V.A.'s query routing engine.
Available folders: [{folders_str}]

If the user's question is about a specific topic that matches a folder, reply ONLY with that folder name.
If it is a general question, coding request, or you are unsure, reply ONLY with "ALL".

Question: {question}
Folder:"""
    
    response = llm(
        prompt,
        max_tokens=10, 
        stop=["\n"], 
        echo=False
    )
    
    answer = response["choices"][0]["text"].strip()
    
    for folder in available_folders:
        if folder.lower() in answer.lower():
            return folder
            
    return "ALL"

def chat_with_context(question: str, db_path: str, model_path: str) -> str:
    llm = get_llm(model_path)

    # 1. Ask the Router what to do
    available_folders = get_available_folders(db_path)
    print(f"[DEBUG] Found folders in DB: {available_folders}")
    
    target_folder = route_query(question, available_folders, llm)
    print(f"[DEBUG] N.O.V.A. routed query to folder: {target_folder}")

    # 2. Perform the Vector Search with our filter
    query_vec = embed_text(question)
    results = search_db(
        query_vec, 
        db_path, 
        return_scores=True,
        target_folder=target_folder
    )

    context_chunks = [doc for doc, _ in results]
    scores = [score for _, score in results]

    print("[DEBUG] Similarity scores:", scores)

    # Check if all context matches are too dissimilar (higher score = less relevant)
    threshold = 1.2
    context_is_relevant = any(score < threshold for score in scores)

    if context_is_relevant and context_chunks:
        prompt = format_context_prompt(question, context_chunks)
    else:
        print("[DEBUG] No relevant context found. Falling back to general knowledge.")
        prompt = format_general_prompt(question)

    print("\n[DEBUG] Prompt being sent to LLM:\n", prompt[:500], "..." if len(prompt) > 500 else "")

    output = llm(prompt, max_tokens=256, stop=["User:", "\n\n"])
    return output["choices"][0]["text"].strip()