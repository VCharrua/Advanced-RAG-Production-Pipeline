from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path = Path("/data/rag-data")
    embedding_model: str = "/models/BAAI--bge-base-en-v1.5"
    reranker_model: str = "/models/BAAI--bge-reranker-base"
    generator_model: str = "/models/Qwen--Qwen2.5-7B-Instruct"
    context_chars: int = 18_000
    max_new_tokens: int = 420
    dense_candidates: int = 48
    broad_dense_candidates: int = 80
    rerank_candidates: int = 32
    broad_rerank_candidates: int = 64

    @property
    def corpus_dir(self) -> Path:
        return self.data_root / "corpus"

    @property
    def chunks_file(self) -> Path:
        return self.corpus_dir / "corpus_chunks.jsonl"

    @property
    def raw_dir(self) -> Path:
        return self.corpus_dir / "raw_documents"
