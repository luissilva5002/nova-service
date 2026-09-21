"""
nova/memory/vector_store.py

Tier 3 memory (Long-Term Knowledge / project & document embeddings).
Wraps ChromaDB so the rest of the app never touches the client
directly. Lives entirely on disk, invisible to the LLM until a query
triggers a JIT (just-in-time) retrieval (see architecture doc,
"Retrieval Cycle").

Falls back to a no-op/in-memory stub if chromadb isn't installed yet,
so the rest of the server can boot and be developed even before the
embedding model / chromadb dependency is fully set up.
"""
from pathlib import Path
from typing import Optional

from nova.config import VECTOR_STORE_DIR, VECTOR_STORE_COLLECTION, RAG_TOP_K, MINILM_MODEL_DIR

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover - allows the server to boot without the dep installed yet
    _CHROMA_AVAILABLE = False


class _LocalEmbeddingFunction:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(str(MINILM_MODEL_DIR), device="cpu", local_files_only=True)

    def __call__(self, input):
        return self._model.encode(input, normalize_embeddings=True).tolist()

    def name(self):
        return "local-minilm"


class VectorStore:
    def __init__(self, persist_dir: Path = VECTOR_STORE_DIR, collection_name: str = VECTOR_STORE_COLLECTION):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        if _CHROMA_AVAILABLE:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            embedding_function = _LocalEmbeddingFunction() if MINILM_MODEL_DIR.exists() else None
            self._collection = self._client.get_or_create_collection(
                self.collection_name, embedding_function=embedding_function,
                metadata={"telemetry": "disabled"},
            )

    @property
    def available(self) -> bool:
        return _CHROMA_AVAILABLE

    def upsert_chunk(self, chunk_id: str, text: str, metadata: dict) -> None:
        """
        metadata should include at least: file_path, project_id, line_start, line_end
        (see architecture doc: "Vector Chunk" + "Metadata Payload").
        """
        if not _CHROMA_AVAILABLE:
            return
        self._collection.upsert(ids=[chunk_id], documents=[text], metadatas=[metadata])

    def upsert_many(self, ids: list, texts: list, metadatas: list) -> None:
        if not _CHROMA_AVAILABLE or not ids:
            return
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, query_text: str, project_id: Optional[str] = None, top_k: int = RAG_TOP_K) -> list:
        """
        Returns the top_k most relevant chunks, each as
        {"text":..., "metadata":..., "distance":...}
        Capped deliberately low (RAG_TOP_K) to keep the second LLM
        prefill fast on CPU (see "Strict RAG Limits" mitigation).
        """
        if not _CHROMA_AVAILABLE:
            return []
        where = {"project_id": project_id} if project_id else None
        result = self._collection.query(query_texts=[query_text], n_results=top_k, where=where)
        hits = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0] if result.get("distances") else [None] * len(docs)
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits

    def delete_project(self, project_id: str) -> None:
        if not _CHROMA_AVAILABLE:
            return
        self._collection.delete(where={"project_id": project_id})


vector_store = VectorStore()
