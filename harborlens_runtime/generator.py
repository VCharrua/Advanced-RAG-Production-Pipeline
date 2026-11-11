import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class GroundedGenerator:
    REFUSAL = "Information not present in context."
    SENTENCE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, config):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.generator_model), local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.generator_model),
            quantization_config=quant,
            device_map="auto",
            local_files_only=True,
        ).eval()

    def _context(self, evidence):
        blocks, used = [], 0
        for row in evidence:
            block = (
                f"[{row['chunk_id']}] {row.get('doc_title', row.get('doc_id', ''))} | "
                f"{row.get('section_title', row.get('section_path', ''))}\n"
                f"{row.get('content', '').strip()}"
            )
            if used + len(block) > self.config.context_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def _fallback(self, evidence):
        if not evidence:
            return self.REFUSAL
        return " ".join(
            f"[{row['chunk_id']}] {sentence.strip()}"
            for row in evidence[:3]
            for sentence in self.SENTENCE.split(row.get("content", "").strip())
            if sentence.strip()
        )

    @staticmethod
    def _numbers_are_grounded(answer, evidence):
        values = lambda text: {
            item.replace(",", "").rstrip(".")
            for item in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
        }
        return values(answer).issubset(values(evidence))

    def _citations_are_valid(self, answer, allowed):
        if answer.strip() == self.REFUSAL:
            return True
        citations = set(re.findall(r"\[(chunk_\d+)\]", answer))
        return bool(citations) and citations.issubset(allowed) and all(
            re.search(r"\[chunk_\d+\]", sentence)
            for sentence in self.SENTENCE.split(answer)
            if sentence.strip()
        )

    def answer(self, query, evidence):
        if not evidence:
            return self.REFUSAL
        evidence_text = self._context(evidence)
        prompt = (
            "Use only the evidence below. Answer precisely. Every factual sentence must "
            f"cite IDs like [chunk_0042]. Preserve units, ranges, dates, and qualifiers. "
            f"If evidence is insufficient, output exactly: {self.REFUSAL}\n"
            f"Question: {query}\nEvidence:\n{evidence_text}\nAnswer:"
        )
        messages = [
            {"role": "system", "content": "You are a citation-first technical analyst. Never use outside knowledge."},
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=7000).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        allowed = {row["chunk_id"] for row in evidence}
        if self._citations_are_valid(answer, allowed) and self._numbers_are_grounded(answer, evidence_text):
            return answer
        return self._fallback(evidence)
