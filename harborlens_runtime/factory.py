import os
from pathlib import Path

import torch

from .config import RuntimeConfig
from .corpus_store import CorpusStore
from .generator import GroundedGenerator
from .retriever import HybridRetriever


def build_pipeline(config=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local HarborLens runtime")
    config = config or RuntimeConfig(
        data_root=Path(os.environ.get("DATA_ROOT", "/data/rag-data")),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "/models/BAAI--bge-base-en-v1.5"),
        reranker_model=os.environ.get("RERANKER_MODEL", "/models/BAAI--bge-reranker-base"),
        generator_model=os.environ.get("GENERATOR_MODEL", "/models/Qwen--Qwen2.5-7B-Instruct"),
    )
    store = CorpusStore(config)
    return HybridRetriever(store, config), GroundedGenerator(config)
