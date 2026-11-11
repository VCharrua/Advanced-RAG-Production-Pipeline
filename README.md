# Advanced RAG Production Pipeline

An offline-first Retrieval-Augmented Generation (RAG) system for the **Advanced Local RAG Benchmark** kaggle competition. The project is designed for mission-critical enterprise information retrieval across naval engineering specifications, contracts, regulatory documents, maintenance manuals, fleet ledgers, and incident records.

The challenge requires a pipeline that can:

- retrieve authoritative canonical passages from heterogeneous technical documents;
- preserve tables, units, clauses, section hierarchy, and cross-references;
- answer single-document and fleet-wide aggregation questions;
- refuse questions that are not supported by retrieved evidence;
- generate answers using local open-weight models only;
- produce a valid `submission.csv` containing the canonical chunk IDs;
- run on Kaggle GPU hardware without commercial cloud APIs.

The full requirements are documented in [`docs/project-specifications.md`](docs/project-specifications.md).

## Repository contents

```text
.
├── docs/
│   └── project-specifications.md
├── harborlens_runtime/
│   ├── config.py
│   ├── corpus_store.py
│   ├── factory.py
│   ├── generator.py
│   └── retriever.py
├── kaggle/
│   └── input/rag-data/
├── Dockerfile
├── harborlens_api.py
├── rag_solution.ipynb
├── rag_solution_v2.ipynb
└── requirements-runtime.txt
```

## Pipeline versions

### `rag_solution.ipynb`

The first independent implementation. It uses:

- structure-aware Markdown enrichment;
- BM25-style lexical retrieval;
- word and character TF-IDF retrieval;
- reciprocal-rank fusion;
- optional local Qwen generation;
- deterministic evidence-only fallback answers;
- submission generation and validation.

This version is lightweight and useful as a portable baseline.

### `rag_solution_v2.ipynb`

The recommended implementation. It extends the baseline with:

- `CorpusStore` for corpus metadata and local indexes;
- `HybridRetriever` for dense, lexical, and reranked retrieval;
- `GroundedGenerator` for local quantized generation and validation;
- `BenchmarkRunner` for reproducible batch execution;
- BGE dense embeddings stored in FAISS;
- BM25 lexical retrieval;
- BGE cross-encoder reranking;
- table-aware and identifier-aware scoring;
- parent-clause, section, and cross-reference metadata;
- adaptive retrieval for aggregation questions;
- lexical abstention for unrelated questions;
- mandatory valid citations in generated answers;
- numeric grounding checks;
- extractive citation-preserving fallbacks;
- OOD/refusal regression probes;
- latency, refusal-rate, and GPU-memory tracing;
- optional Recall@K, Precision@K, and MRR evaluation;
- Docker and FastAPI packaging output.

Use `rag_solution_v2.ipynb` for the benchmark submission and engineering review.

## Dataset

The supplied corpus is mounted at:

```text
/kaggle/input/rag-data/
├── corpus/
│   ├── corpus_chunks.jsonl
│   ├── corpus_manifest.json
│   └── raw_documents/
└── test_suite/
    └── test_queries.csv
```

The canonical `corpus_chunks.jsonl` file contains the official `chunk_XXXX` identifiers used for retrieval scoring. The notebook preserves these IDs in the `retrieved_chunk_ids` submission column.

## Kaggle execution

Configure the Kaggle notebook with:

1. GPU accelerator enabled, preferably T4 or P100.
2. Internet enabled only for package and model provisioning.
3. The `rag-data` dataset attached.
4. The selected notebook uploaded or copied into the Kaggle session.

Run all cells from top to bottom. The notebook creates:

```text
submission.csv
retrieval_trace.csv
harborlens_manifest.json
```

The submission must contain exactly:

```text
query_id,retrieved_chunk_ids,generated_answer
```

The final validation cell checks:

- exactly 200 rows;
- unique query IDs;
- exact query ID coverage;
- canonical retrieved chunk IDs;
- non-empty generated-answer output.

## Local runtime package

The `harborlens_runtime` package separates deployment code from notebook state:

- `RuntimeConfig` centralizes paths and model configuration.
- `CorpusStore` loads raw/canonical documents and builds the dense and BM25 indexes.
- `HybridRetriever` performs fusion, entity boosts, reranking, abstention, and evidence selection.
- `GroundedGenerator` loads the local quantized model and validates citations/numerical claims.
- `build_pipeline()` creates the services used by the API.

The runtime expects local model directories:

```text
/models/BAAI--bge-base-en-v1.5
/models/BAAI--bge-reranker-base
/models/Qwen--Qwen2.5-7B-Instruct
```

The models must be provisioned before enabling offline mode.

## Docker deployment

The [Dockerfile](Dockerfile) uses an NVIDIA CUDA runtime image because embeddings, reranking, and generation run on the GPU. It installs CUDA-compatible PyTorch separately from the remaining pinned dependencies in [`requirements-runtime.txt`](requirements-runtime.txt).

Build:

```powershell
docker build -t harborlens:latest .
```

Run with the models mounted read-only:

```powershell
docker run --gpus all `
  -v C:\models:/models:ro `
  -p 8000:8000 `
  harborlens:latest
```

The API exposes:

```text
GET  /health
POST /answer
```

Example request:

```json
{
  "query": "What is the continuous maximum power rating of the GTX-9000?",
  "category": "Category A"
}
```

The response includes the grounded answer, canonical chunk IDs, and request latency.

The container intentionally runs one Uvicorn worker. Multiple workers would duplicate the local language model in GPU memory. Horizontal scaling should use one container per GPU.

## Evaluation and rubric alignment

The implementation targets the `Production Engineering Review Rubric` in the project specification.

### RAG architecture and vector indexing

- hierarchical Markdown section enrichment;
- parent-clause and cross-reference preservation;
- table and structured-data metadata;
- canonical chunk ID preservation;
- dense FAISS search;
- BM25 lexical search;
- reciprocal-rank fusion;
- cross-encoder reranking;
- aggregation-aware candidate breadth;
- diversity-aware evidence selection;
- dynamic abstention.

### Generation and hallucination control

- local Qwen open-weight model;
- 4-bit NF4 quantization;
- deterministic decoding;
- strict evidence-only prompting;
- mandatory valid chunk citations;
- numeric claim verification;
- extractive citation-preserving fallback;
- out-of-domain refusal probes.

### Engineering and MLOps

- class-based native architecture;
- centralized configuration and deterministic seeds;
- reusable runtime package;
- FastAPI service;
- CUDA Docker image;
- pinned runtime dependencies;
- non-root container user;
- latency and GPU-memory tracing;
- refusal and retrieval observability;
- manifest generation;
- submission validation;
- optional labeled retrieval metrics.

## Security and operating constraints

The project does not call OpenAI, Anthropic, Google Gemini, Cohere, or any other commercial inference API. All embedding, reranking, and generation operations are local.

Internet access is only needed during dependency/model provisioning. For an air-gapped deployment:

1. download and verify model artifacts before deployment;
2. mount them into `/models`;
3. keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
4. build or transfer the image into the enclave;
5. expose the API only on the approved internal network.

## Development checks

The notebooks and runtime modules can be statically checked without loading GPU models:

```powershell
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('harborlens_runtime').glob('*.py')]"
```

Full retrieval and generation validation requires the Kaggle GPU environment or an equivalent CUDA-enabled Docker runtime.
