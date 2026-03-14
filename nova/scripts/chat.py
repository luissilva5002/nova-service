from llama_cpp import Llama
from nova.vector.embed import embed_text
from nova.vector.db import search_db

# Global cache for model instance
_llm = None

def get_llm(model_path: str = "models/nova_model_lora.gguf"):
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=6,
            verbose=False
        )
    return _llm

def format_context_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
    return f"""You are N.O.V.A., an intelligent assistant.

Answer the following question based only on the context provided.

Context:
{context}

Question: {question}
Answer:"""

def format_general_prompt(question: str) -> str:
    return f"""You are N.O.V.A., an intelligent assistant.

You don't have any relevant context for this question, so answer it using your general knowledge.

Question: {question}
Answer:"""

def chat_with_context(question: str, db_path: str, model_path: str) -> str:
    llm = get_llm(model_path)

    query_vec = embed_text(question)
    results = search_db(query_vec, db_path, return_scores=True)

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