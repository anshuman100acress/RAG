from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="clinic_faq")

def ingest_markdown(file_path: str) -> None:
    text = Path(file_path).read_text(encoding="utf-8")
    # Split on "## " headers so each policy stays in one chunk
    sections = [s.strip() for s in text.split("## ") if s.strip()]
    chunks = [f"## {s}" for s in sections]

    embeddings = embed_model.encode(chunks, normalize_embeddings=True)

    collection.upsert(
        ids=[f"faq_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": file_path, "section": c.splitlines()[0]} for c in chunks],
    )
    print(f"Indexed {len(chunks)} clinic policies into ChromaDB.")

if __name__ == "__main__":
    ingest_markdown("data/clinic_faq.md")