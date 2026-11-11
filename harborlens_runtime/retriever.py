import re

import numpy as np


class HybridRetriever:
    AGGREGATE = re.compile(
        r"\b(all|every|each|across|total|aggregate|fleet|trend|between\s+\d{4}|"
        r"how many|count|compare|combined|sum|list)\b",
        re.I,
    )
    ENTITY = re.compile(
        r"[A-Z]{2,}[A-Z0-9]*(?:[-_/][A-Z0-9]+)+|"
        r"\b\d+(?:\.\d+)?(?:%|\s?(?:MW|bar|RPM|months?|years?|days?|kg/s|°C|mm/s))?",
        re.I,
    )
    STOP = set("the a an and or is are was were what which how does do of for to in on with from this that be as it by".split())

    def __init__(self, store, config):
        self.store = store
        self.config = config

    def is_broad(self, query, category):
        return bool(self.AGGREGATE.search(query)) or "category e" in str(category).lower() or len(query.split()) > 24

    @staticmethod
    def _rrf(order, size, depth=60, offset=60):
        scores = np.zeros(size, dtype=np.float32)
        for rank, index in enumerate(order[:depth]):
            scores[index] += 1 / (offset + rank + 1)
        return scores

    def _candidates(self, query, category):
        terms = {term for term in self.store.tokenize(query) if term not in self.STOP and len(term) > 1}
        if terms and not terms.intersection(self.store.vocab):
            return []
        broad = self.is_broad(query, category)
        dense_limit = self.config.broad_dense_candidates if broad else self.config.dense_candidates
        _, dense_indices = self.store.faiss_index.search(
            self.store.encode_query(query), dense_limit
        )
        lexical = self.store.bm25.get_scores(self.store.tokenize(query))
        fused = (
            0.62 * self._rrf(dense_indices[0], len(self.store.passages), dense_limit)
            + 0.38 * self._rrf(np.argsort(-lexical), len(self.store.passages), 100)
        )
        query_entities = {item.lower() for item in self.ENTITY.findall(query)}
        for index, row in enumerate(self.store.passages):
            row_entities = {item.lower() for item in self.ENTITY.findall(row.get("content", ""))}
            if query_entities.intersection(row_entities):
                fused[index] += 0.035
            if broad and row.get("doc_type") in {"fleet_incident_log", "maintenance_ledger"}:
                fused[index] += 0.012
            if row.get("is_table") and any(term in row.get("retrieval_text", "").lower() for term in terms):
                fused[index] += 0.01
        count = self.config.broad_rerank_candidates if broad else self.config.rerank_candidates
        return [(int(index), float(fused[index])) for index in np.argsort(-fused)[:count]]

    def retrieve(self, query, category=""):
        candidates = self._candidates(query, category)
        if not candidates:
            return []
        pairs = [(query, self.store.passages[index]["retrieval_text"]) for index, _ in candidates]
        rerank_scores = self.store.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = sorted(
            [
                (index, 0.72 * float(rerank) + 0.28 * fusion)
                for (index, fusion), rerank in zip(candidates, rerank_scores)
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked or ranked[0][1] < -1.5:
            return []
        maximum = 12 if self.is_broad(query, category) else 6
        selected, seen = [], set()
        for index, score in ranked:
            row = self.store.passages[index]
            key = (row.get("doc_id", ""), row.get("parent_clause_id", "") or row.get("section_id", ""))
            if not selected or key not in seen or self.is_broad(query, category):
                selected.append({**row, "rerank_score": float(score), "rank": len(selected) + 1})
                seen.add(key)
            if len(selected) >= maximum:
                break
        return selected
