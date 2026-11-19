from contextlib import asynccontextmanager
import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from agent import ClinicDeps, ReceptionistResponse, emma_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load heavy models and DB connections once
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="clinic_faq")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    app.state.deps = ClinicDeps(collection=collection, embed_model=embed_model)
    yield

app = FastAPI(title="Mini-EMMA GP Receptionist API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health_check(request: Request):
    count = request.app.state.deps.collection.count()
    return {"status": "ok", "indexed_policies": count}

@app.post("/upload-faq")
async def upload_faq(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only Markdown (.md) files are supported.")

    text = (await file.read()).decode("utf-8")
    sections = [s.strip() for s in text.split("## ") if s.strip()]
    chunks = [f"## {s}" for s in sections]

    if not chunks:
        raise HTTPException(status_code=400, detail="No '##' sections found in file.")

    deps: ClinicDeps = request.app.state.deps
    embeddings = deps.embed_model.encode(chunks, normalize_embeddings=True)

    deps.collection.upsert(
        ids=[f"{file.filename}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": file.filename, "section": c.splitlines()[0]} for c in chunks],
    )
    return {"status": "indexed", "filename": file.filename, "chunks_added": len(chunks)}

@app.post("/chat", response_model=ReceptionistResponse)
async def chat(payload: ChatRequest, request: Request):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    deps: ClinicDeps = request.app.state.deps
    # Use async .run() inside FastAPI async routes
    result = await emma_agent.run(payload.message, deps=deps)
    return result.output
