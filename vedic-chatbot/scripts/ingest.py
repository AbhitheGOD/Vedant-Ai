"""
CLI script to ingest Vedic texts into ChromaDB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --reset   # Clear existing data before ingesting
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import rag_service
from app.services.ingestion_service import load_all_documents


def main():
    parser = argparse.ArgumentParser(description="Ingest Vedic texts into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Clear existing ChromaDB data before ingesting")
    args = parser.parse_args()

    print("=" * 60)
    print("Vedic Knowledge Base Ingestion")
    print("=" * 60)

    # Initialize RAG service (loads embedding model + ChromaDB)
    rag_service.initialize()

    if args.reset:
        print("\nResetting ChromaDB collection...")
        from app.config import settings
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(rag_service.COLLECTION_NAME)
            print("Collection deleted.")
        except Exception:
            pass
        rag_service.initialize()

    print(f"\nLoading documents from data/texts/...")
    texts, metadatas = load_all_documents()

    if not texts:
        print("No documents found. Make sure data/texts/ contains .txt or .json files.")
        sys.exit(1)

    print(f"\nInserting {len(texts)} chunks into ChromaDB...")
    added = rag_service.add_documents(texts, metadatas)

    print(f"\nIngestion complete!")
    print(f"  New chunks added : {added}")
    print(f"  Total in DB      : {rag_service.get_document_count()}")
    print(f"  Sources          : {', '.join(rag_service.get_available_sources())}")
    print("\nYou can now start the server: python run.py")


if __name__ == "__main__":
    main()
