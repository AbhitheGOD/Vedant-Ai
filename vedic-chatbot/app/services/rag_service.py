from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models.schemas import SourceInfo


class RAGService:
    COLLECTION_NAME = "vedic_knowledge"

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None

    def initialize(self):
        """Load embedding model and initialize ChromaDB. Called once at startup."""
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        print(f"Initializing ChromaDB at: {settings.CHROMA_PERSIST_DIR}")
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB ready. Documents in collection: {self.collection.count()}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def add_documents(self, texts: List[str], metadatas: List[Dict]) -> int:
        """Add chunked texts with metadata. Returns number of new docs added."""
        if not texts:
            return 0

        # Build IDs deterministically for deduplication
        existing_ids = set(self.collection.get()["ids"])
        new_texts, new_metas, new_ids = [], [], []

        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            doc_id = f"{meta.get('source', 'unknown')}_{meta.get('chunk_index', i)}"
            if doc_id not in existing_ids:
                new_texts.append(text)
                new_metas.append(meta)
                new_ids.append(doc_id)

        if not new_texts:
            return 0

        embeddings = self._embed(new_texts)
        # Batch insert in chunks of 500 to avoid memory issues
        batch_size = 500
        for start in range(0, len(new_texts), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=new_ids[start:end],
                embeddings=embeddings[start:end],
                documents=new_texts[start:end],
                metadatas=new_metas[start:end],
            )

        return len(new_texts)

    def query(self, text: str, top_k: int = None) -> List[SourceInfo]:
        """Return top_k most relevant chunks above the relevance threshold."""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS

        if self.collection.count() == 0:
            return []

        query_embedding = self._embed([text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        sources = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # cosine distance -> similarity: similarity = 1 - distance
            similarity = 1.0 - distance
            if similarity >= settings.RELEVANCE_THRESHOLD:
                sources.append(SourceInfo(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chapter=meta.get("chapter", ""),
                    verse=meta.get("verse", ""),
                    relevance_score=round(similarity, 4),
                ))

        return sources

    def get_document_count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()

    def get_available_sources(self) -> List[str]:
        if self.collection is None or self.collection.count() == 0:
            return []
        all_docs = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in all_docs["metadatas"]}
        return sorted(sources)


rag_service = RAGService()
