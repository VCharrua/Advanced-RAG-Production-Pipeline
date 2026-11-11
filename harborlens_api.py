import time

from fastapi import FastAPI
from pydantic import BaseModel

from harborlens_runtime import build_pipeline


retriever, generator = build_pipeline()
app = FastAPI(title="HarborLens Pro Local RAG")


class Query(BaseModel):
    query: str
    category: str = ""


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/answer")
def answer(request: Query):
    started = time.perf_counter()
    evidence = retriever.retrieve(request.query, request.category)
    response = generator.answer(request.query, evidence)
    return {
        "answer": response,
        "chunk_ids": [row["chunk_id"] for row in evidence],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
