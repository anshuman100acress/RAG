from dataclasses import dataclass
from typing import Literal
import chromadb
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sentence_transformers import SentenceTransformer

# 1. Define the strict structured output for voice & routing
class ReceptionistResponse(BaseModel):
    spoken_response: str = Field(
        description="Short, polite 1-2 sentence reply suitable for Text-to-Speech."
    )
    intent: Literal["booking", "prescription", "faq", "emergency", "human_handoff"]
    escalate_to_human: bool = Field(
        description="True if the caller needs a human receptionist or has a medical issue."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Section titles retrieved from the clinic FAQ."
    )

# 2. Define runtime dependencies
@dataclass
class ClinicDeps:
    collection: chromadb.Collection
    embed_model: SentenceTransformer

# 3. Create the PydanticAI agent
emma_agent = Agent(
    "google:gemini-3.8-flash",
    deps_type=ClinicDeps,
    output_type=ReceptionistResponse,
    instructions=(
        "You are EMMA, a virtual receptionist for Riverside GP Surgery. "
        "ALWAYS use the `search_clinic_faq` tool before answering clinic policy questions. "
        "Never invent policies—if the FAQ does not cover it, set escalate_to_human=True. "
        "CLINICAL SAFETY: Never give medical advice or diagnose symptoms. "
        "If a caller mentions chest pain, difficulty breathing, or a medical emergency, "
        "immediately set intent='emergency', escalate_to_human=True, and tell them to call 999 or go to A&E."
    ),
)

# 4. Register the ChromaDB retriever as a tool
@emma_agent.tool
def search_clinic_faq(ctx: RunContext[ClinicDeps], query: str) -> str:
    """Search Riverside GP Surgery policies and FAQs."""
    query_vec = ctx.deps.embed_model.encode([query], normalize_embeddings=True)
    results = ctx.deps.collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=2,
    )
    docs = results["documents"][0]
    return "\n\n---\n\n".join(docs) if docs else "No relevant policy found."

# 5. Test the 3 critical interview scenarios
if __name__ == "__main__":
    deps = ClinicDeps(
        collection=chromadb.PersistentClient(path="./chroma_db").get_collection("clinic_faq"),
        embed_model=SentenceTransformer("BAAI/bge-small-en-v1.5"),
    )

    test_queries = [
        "Can I order my repeat prescription over the phone?",
        "Do you offer free parking behind the clinic?",
        "I'm having severe chest pain and feel dizzy.",
    ]

    for q in test_queries:
        result = emma_agent.run_sync(q, deps=deps)
        print(f"\nCaller: {q}")
        print(result.output.model_dump_json(indent=2))