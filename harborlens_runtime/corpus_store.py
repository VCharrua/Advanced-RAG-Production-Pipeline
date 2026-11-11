import json
import re
from collections import defaultdict

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class CorpusStore:
    def __init__(self, config):
        self.config = config
        with config.chunks_file.open(encoding="utf-8") as handle:
            source = [json.loads(line) for line in handle if line.strip()]
        raw_docs = {
            path.stem: path.read_text(encoding="utf-8")
            for path in config.raw_dir.glob("*.md")
        }
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
        headings = {
            key: [(m.start(), len(m.group(1)), m.group(2).strip())
                   for m in heading_re.finditer(body)]
            for key, body in raw_docs.items()
        }

        def locate(doc_id, text):
            body = raw_docs.get(doc_id, "")
            probe = text.strip()[:180]
            position = body.find(probe)
            if position >= 0:
                return position
            match = re.search(r"\s+".join(re.escape(part) for part in probe.split()), body)
            return match.start() if match else None

        def section_path(doc_id, position):
            stack = []
            for start, depth, title in headings.get(doc_id, []):
                if start > position:
                    break
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                stack.append((depth, title))
            return " > ".join(title for _, title in stack)

        passages = []
        for row in source:
            content = row.get("content", row.get("text", ""))
            position = locate(row.get("doc_id", ""), content)
            section = (
                section_path(row.get("doc_id", ""), position)
                if position is not None
                else row.get("section_title", "")
            )
            fields = [
                row.get("doc_id", ""),
                row.get("doc_type", ""),
                section,
                row.get("parent_clause_id", ""),
            ]
            if row.get("cross_references"):
                fields.append("cross references: " + " ".join(map(str, row["cross_references"])))
            if row.get("is_table"):
                fields.append("TABLE STRUCTURE")
            if row.get("structured_data"):
                fields.append("table fields: " + json.dumps(
                    row["structured_data"], sort_keys=True, ensure_ascii=False
                ))
            passages.append({
                **row,
                "section_path": section,
                "retrieval_text": " | ".join(filter(None, fields)) + "\n" + content,
            })

        self.passages = passages
        self.chunk_ids = [row["chunk_id"] for row in passages]
        self.embedder = SentenceTransformer(
            str(config.embedding_model), device="cuda", local_files_only=True
        )
        vectors = self.embedder.encode(
            [row["retrieval_text"] for row in passages],
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss_index.add(vectors.astype("float32"))
        token_re = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*")
        self.tokenize = lambda text: token_re.findall(text.lower())
        tokenized = [self.tokenize(row["retrieval_text"]) for row in passages]
        self.vocab = {token for row in tokenized for token in row}
        self.bm25 = BM25Okapi(tokenized, k1=1.35, b=0.72)
        self.reranker = CrossEncoder(
            str(config.reranker_model),
            device="cuda",
            max_length=512,
        )

    def encode_query(self, query):
        return self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
