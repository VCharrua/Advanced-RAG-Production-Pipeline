

# Advanced Local RAG Benchmark: Production-Grade Systems

Build an air-gapped RAG pipeline using local open-weight LLMs to overcome complex enterprise document failure modes without hallucinating.

## Overview

### The Mission

In mission-critical enterprise domains—such as naval propulsion engineering, defense procurement contracting, maritime regulatory compliance, and fleet maintenance—information retrieval cannot afford generic, naive RAG pipelines. Standard recursive character splitters, simple cosine vector lookups, and fixed top-k context stuffing predictably fail when confronted with:

*   Nested contractual subclauses disconnected from parent clauses.
*   Complex multi-attribute engineering tables and financial ledgers.
*   Fleet-wide aggregation queries requiring synthesis across 30 to 100+ documents.
*   Subtle numerical discrepancies across identical maintenance procedures.  
    The objective of this competition is to **design, implement, and deploy a production-grade, 100% air-gapped Local RAG pipeline** operating over a **simulated enterprise technical corpus across 20 document suites**.  
    Your system must extract facts, cross-reference clauses, aggregate cross-document evidence, preserve tabular data integrity, and generate faithful, authoritative answers—**running completely offline on Kaggle hardware without external APIs**.


## Description

### The Engineering Challenge

Your pipeline will be evaluated under strict engineering and architectural constraints simulating an on-premise defense or enterprise enclave.

#### Critical System Constraints & Execution Rules

*   **Execution Platform:** Must run within Kaggle's standard free-tier resources (single session limit, T4/P100 GPU, default RAM/disk quotas).
    *   **Memory & Resource Management:** Pipeline must manage batching and memory cleanup properly to avoid Out-of-Memory (OOM) crashes or runtime timeouts.
*   **Internet Access Policy (Enabled for Dependency & Model Provisioning):**
    *   **Internet is ENABLED** in your Kaggle notebook to allow standard package installation (`pip install sentence-transformers langchain llama-index faiss-gpu`) and direct loading of open-weight models from Hugging Face or Kaggle Models.
*   **STRICT PROHIBITION: No Commercial Cloud APIs:**
    *   **Zero External API Calls:** You are strictly prohibited from calling commercial LLM/embedding APIs (OpenAI, Anthropic, Google Gemini, Cohere, etc.).
    *   **Local Models Only:** All embeddings, reranking, and generation must execute directly on the Kaggle GPU hardware using open-weight models with a parameter footprint of **```$\le 9\text{B}$```** (e.g. Llama-3-8B, Mistral-7B, Qwen-2.5-7B, Gemma-2-9B).
    *   **Enforcement & Disqualification:** Every submission is audited. Calling any commercial API endpoint or having zero GPU VRAM utilization during generation triggers **immediate disqualification**.

### The Competition Dataset

The dataset is mounted under `/kaggle/input/rag-data/` and comprises **20 authentic enterprise document suites** reflecting real-world engineering specifications, procurement contracts, maritime regulatory frameworks, fleet maintenance logs, and financial ledgers:

    /kaggle/input/rag-data/
    ├── corpus/
    │   ├── raw_documents/                       # Authentic Markdown source documents
    │   │   ├── DEF-NAV-CTR-2023-882.md
    │   │   ├── NAV-SPEC-GTX-9000.md
    │   │   ├── FLEET-MAINT-LEDGER.md
    │   │   └── ...
    │   ├── corpus_chunks.jsonl                  # Canonical passages with persistent deterministic IDs ('chunk_XXXX')
    │   └── corpus_manifest.json                 # Corpus metadata manifest
    └── test_suite/
       └── test_queries.csv                     # Evaluation query suite
    

#### Dataset Components

1.  **`raw_documents/`**: Full, un-chunked authentic Markdown source documents. Use these if you want to implement custom parsing, hierarchical chunking, or metadata linking.
2.  **`corpus_chunks.jsonl`**: Canonical pre-extracted passages with persistent IDs (`chunk_XXXX`). These IDs serve as the official retrieval scoring targets.
3.  **`corpus_manifest.json`**: Metadata summary describing document suites, document types, and file structure across the corpus.
4.  **`test_queries.csv`**: The evaluation queries (`query_id`, `category`, `query_text`) to evaluate your RAG pipeline against.

#### How Chunk IDs Are Required for Test Queries

For each query in `test_queries.csv`, your submission must output the matching canonical chunk IDs from `corpus_chunks.jsonl` in the `retrieved_chunk_ids` column:

*   Format: pipe-delimited string (e.g. `chunk_0042|chunk_0043`), ordered by relevance (most relevant first for MAP@K precision).
*   If you chunk `raw_documents/` using your own custom strategy, you must map your retrieved text spans back to their corresponding canonical `chunk_XXXX` IDs for submission.


## Submission Requirements

All participants submit their work as a **Kaggle Writeup & Submission Package** consisting of three interconnected components:

### 1\. The Submission CSV File (`submission.csv`)

*   Generated by running your local pipeline over all 200 test scenarios in `/kaggle/input/rag-data/test_suite/test_queries.csv`.
*   Must be submitted separately / attached alongside your writeup.
*   Exactly 200 rows + 1 header row (`query_id,retrieved_chunk_ids,generated_answer`):

     query_id,retrieved_chunk_ids,generated_answer
     Q_CAT_A_001,chunk_0001,"The GTX-9000 delivers a continuous maximum rating of 36.5 MW with 41.8% thermal efficiency."
     Q_CAT_B_001,chunk_0055|chunk_0056,"Section 5.1 specifies a 30-day cure period with liquidated damages capped at 15% per Appendix B."
     Q_CAT_E_001,chunk_0081|chunk_0084|chunk_0087|...,"Between 2021 and 2024, exactly 32 structural hull micro-fractures occurred..."
    

#### Detailed Column Specifications:

*   `query_id`: Exact string matching the 200 query identifiers in `test_queries.csv` (e.g. `Q_CAT_A_001` to `Q_CAT_E_040`). All 200 queries must be present.
*   `retrieved_chunk_ids`: Pipe-delimited string of retrieved canonical chunk IDs (e.g. `chunk_0055|chunk_0056`).
    *   **Ordering Matters:** List chunks in order of relevance; MAP@K precision awards higher scores to relevant chunks appearing near the beginning of the list.
    *   **Empty Retrieval:** If your system determines a query is unanswerable or finds no relevant context, leave this field empty `""` (do not put synthetic placeholders).
    *   **ID Normalization:** IDs should match `corpus_chunks.jsonl` (e.g. `chunk_0042`).
*   `generated_answer`: The final textual response generated by your local LLM. Multi-line answers or answers containing commas must be standard CSV-escaped using double quotes `"` around the cell.

### 2\. Executable Public Kaggle Notebook (Code Submission)

*   Attached directly to your Kaggle Writeup submission.
*   Must execute on Kaggle GPU hardware (`GPU T4 x2` or `GPU P100`) using local open-weight models.
*   Demonstrates reproducible document parsing, vector indexing, retrieval, and local open-weight LLM generation. Calling external cloud APIs is strictly prohibited.
*   Self-contained and runnable from top to bottom ("Save & Run All").

### 3\. System Architecture & Technical Report (The Writeup Body)

A comprehensive markdown engineering report detailing:

*   **Framework & Pipeline Choice:** Rationale for choosing **LangChain**, **LlamaIndex**, **LangGraph**, custom native pipelines, etc.
*   **Document Ingestion & Indexing Strategy:** Architectural design for handling heterogeneous documents (contracts, tables, manuals), structure preservation, metadata management, and local vector indexing.
*   **Retrieval Strategy & Optimization:** Techniques employed to maximize retrieval recall and precision across complex multi-document and high-volume queries, candidate re-scoring/re-ranking, and dynamic thresholding.
*   **Generation Fidelity & Hallucination Control:** Approaches used to enforce strict factual grounding, citation accuracy, unanswerable query refusal, and prompt engineering strategies.
*   **Production Deployment & MLOps Blueprint:** Architectural design for exposing the RAG pipeline as a high-throughput, containerized local API service (e.g. FastAPI / vLLM) in an offline enterprise enclave, with CI/CD regression testing, logging, monitoring, and fallback heuristics.

### 4\. Submission Validation Snippet (Run in Notebook)

Before uploading, paste and run this automated validation check at the end of your Kaggle notebook:

```python
import pandas as pd
# Load candidate submission
sub = pd.read_csv("submission.csv")
# 1. Check columns
expected_cols = ["query_id", "retrieved_chunk_ids", "generated_answer"]
assert list(sub.columns) == expected_cols, f"Columns mismatch! Expected {expected_cols}, got {list(sub.columns)}"
# 2. Check row count
assert len(sub) == 200, f"Expected exactly 200 query rows, found {len(sub)}"
# 3. Check for missing query IDs
assert not sub["query_id"].isnull().any(), "Found NULL values in query_id column!"
assert len(sub["query_id"].unique()) == 200, "Duplicate query_id entries detected!"
# 4. Check answers exist
assert not sub["generated_answer"].isnull().all(), "Generated answers are completely empty!"
print("[SUCCESS] submission.csv is 100% compliant and ready for upload!")
```

## Tracks and Awards

### Main Track

*   trophy
    
    ###### GenAI Engineering Excellence Recognition
    
    This recognition acknowledges the candidate demonstrating outstanding technical capability in Generative AI engineering through this assessment. Candidates are evaluated on RAG architecture, retrieval effectiveness, generation quality, hallucination control, offline model deployment, evaluation methodology, and software engineering practices. This recognition is intended for assessment and candidate evaluation purposes and does not represent a monetary or physical prize.


## Evaluation

Submissions are evaluated through a dual-track assessment: **Automated Benchmark Leaderboard Scoring** and **Expert Engineering Writeup Review**. Both your notebook submission (`submission.csv`) and your technical writeup will be used together to validate and rank your solution.

### Evaluation Workflow

1.  **Automated Submission Scoring:**  
    Your submitted `submission.csv` is evaluated across both retrieval precision/recall and generation fidelity.
2.  **Qualitative Engineering Writeup Review:**  
    The engineering panel reviews the attached Kaggle Notebook and evaluates the Technical Writeup against the **Production Engineering Review Rubric**.

#### 1\. Retrieval Metrics

*   **Context Recall:** Proportional coverage of canonical ground-truth chunk IDs  

*   **Context Precision (MAP@K):** Evaluates rank position of relevant documents over retrieved candidates  

*   **Retrieval Composite F1:** Combines precision and recall into a single metric.

#### 2\. Generation Metrics

*   **Faithfulness:** Verifies that claims in the generated response are substantiated strictly by the candidate's retrieved passages without ungrounded parametric priors.
*   **Answer Relevancy:** Measures technical and semantic alignment against authoritative references, incorporating unigram recall, key numerical/entity precision, and bigram phrase continuity.
*   **Generation Composite:**


### Production Engineering Review Rubric (100 Points)

Aligned with the **GenAI Engineer Competency Matrix**:

| Category | Max Points | Evaluation Criteria |
| --- | --- | --- |
| **1\. RAG Architecture & Vector Indexing** | **35 Points** | \- **Data Ingestion & Indexing Quality (10 pts):** Effective handling of complex document structures, tables, hierarchical sections, and metadata enrichment. <BR/> \- **Retrieval Architecture & Precision (15 pts):** Soundness of retrieval mechanics, balancing lexical/semantic recall and precision over local vector storage. <BR/> \- **Advanced Retrieval Innovations (10 pts):** Sophistication in candidate re-scoring, handling broad multi-document queries, and dynamic thresholding. |
| **2\. LLM Generation & Hallucination Control** | **35 Points** | \- **Local LLM Execution (10 pts):** Efficient offline inference of open-weights (Llama-3-8B, Mistral-7B, Qwen-2.5-7B) using 4-bit/8-bit quantization (AWQ/GPTQ/GGUF). <BR/> \- **Faithfulness & Grounding (15 pts):** Strict contextual grounding; responses directly cite retrieved clauses without external hallucinations. <BR/> \- **Adversarial & Edge-Case Handling (10 pts):** Gracefully refusing unanswerable queries ("Information not present in context") rather than confabulating. |
| **3\. Engineering Quality & Production MLOps** | **30 Points** | \- **Framework Mastery (10 pts):** Clean, modular code leveraging LangChain, LlamaIndex, LangGraph, or native architectures with clear abstractions. <BR/> \- **On-Premise Deployment Architecture (10 pts):** Production-ready API service design (FastAPI / vLLM), containerization strategy, batching logic. <BR/> \- **Observability & Fallbacks (10 pts):** Logging frameworks, automated regression testing harnesses, latency monitoring, and fallback heuristics. |
