"""Retrieval and lightweight answer synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from langchain_core.documents import Document

from .config import AppConfig
from .vector_store import VectorStoreManager


CHINESE_STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "是否",
    "一个",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "以及",
    "进行",
    "有关",
    "哪些",
    "可以",
    "需要",
}

ENGLISH_STOPWORDS = {
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
}


@dataclass
class AnswerResult:
    """Structured answer payload returned to the UI."""

    answer: str
    sources: List[Dict[str, str]]
    retrieved_chunks: List[Dict[str, str]]


class QAEngine:
    """Run retrieval and synthesize an answer from evidence chunks."""

    def __init__(self, config: AppConfig, vector_store_manager: VectorStoreManager) -> None:
        self.config = config
        self.vector_store_manager = vector_store_manager

    def ask(self, question: str) -> AnswerResult:
        """Retrieve related chunks and build an evidence-grounded answer."""
        question = question.strip()
        if not question:
            return AnswerResult(
                answer="请输入一个具体问题，我会基于知识库内容回答。",
                sources=[],
                retrieved_chunks=[],
            )

        search_results = list(self.vector_store_manager.similarity_search(question, self.config.top_k))
        if not search_results:
            return AnswerResult(
                answer="当前还没有可用的知识库，请先在左侧上传 PDF 或 TXT 文档并构建向量索引。",
                sources=[],
                retrieved_chunks=[],
            )

        answer = self._synthesize_answer(question, search_results)
        sources = self._build_source_cards(search_results)
        retrieved_chunks = self._build_chunk_cards(search_results)
        return AnswerResult(answer=answer, sources=sources, retrieved_chunks=retrieved_chunks)

    def _synthesize_answer(
        self,
        question: str,
        search_results: Sequence[Tuple[Document, float]],
    ) -> str:
        """Create a concise answer using the highest-value retrieved sentences."""
        keywords = self._extract_keywords(question)
        sentence_candidates = []

        for rank, (document, distance) in enumerate(search_results, start=1):
            sentences = self._split_sentences(document.page_content)
            doc_weight = 1 / (1 + float(distance))

            for sentence_index, sentence in enumerate(sentences):
                normalized_sentence = sentence.strip()
                if len(normalized_sentence) < 12:
                    continue

                keyword_hits = sum(1 for keyword in keywords if keyword in normalized_sentence.lower())
                length_bonus = 0.4 if 20 <= len(normalized_sentence) <= 120 else 0.0
                rank_bonus = max(0.0, 0.3 - (rank - 1) * 0.05)
                position_bonus = 0.2 if sentence_index < 2 else 0.0

                score = doc_weight * 3.0 + keyword_hits * 1.6 + length_bonus + rank_bonus + position_bonus
                sentence_candidates.append(
                    {
                        "text": normalized_sentence,
                        "score": score,
                        "source": document.metadata.get("source", "Unknown"),
                    }
                )

        unique_sentences = []
        seen_sentences = set()

        for item in sorted(sentence_candidates, key=lambda value: value["score"], reverse=True):
            normalized = re.sub(r"\s+", " ", item["text"]).strip().lower()
            if normalized in seen_sentences:
                continue

            seen_sentences.add(normalized)
            unique_sentences.append(item)

            if len(unique_sentences) >= self.config.max_answer_sentences:
                break

        if not unique_sentences:
            top_document = search_results[0][0]
            preview = top_document.page_content[:180].strip()
            return (
                "我检索到了相关文档，但没有找到足够集中的句子来直接回答。"
                f"最相关的内容片段是：{preview}"
            )

        answer_lines = ["基于知识库检索结果，可以这样回答："]

        for index, item in enumerate(unique_sentences, start=1):
            answer_lines.append(f"{index}. {item['text']} [{item['source']}]")

        answer_lines.append("如果你愿意，我还可以继续追问、总结重点，或者按面试风格重写答案。")
        return "\n".join(answer_lines)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split chunk text into smaller sentence-like units."""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        parts = re.split(r"(?<=[。！？!?；;.\n])\s*", text)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        """Extract rough Chinese and English keywords without extra dependencies."""
        lowered = question.lower()
        english_tokens = [
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", lowered)
            if token not in ENGLISH_STOPWORDS
        ]

        chinese_only = "".join(re.findall(r"[\u4e00-\u9fff]", question))
        chinese_tokens = []
        for size in (3, 2):
            for index in range(0, max(len(chinese_only) - size + 1, 0)):
                token = chinese_only[index : index + size]
                if token not in CHINESE_STOPWORDS:
                    chinese_tokens.append(token)

        unique_keywords = []
        seen = set()

        for token in english_tokens + chinese_tokens:
            if token in seen:
                continue
            seen.add(token)
            unique_keywords.append(token)

        return unique_keywords

    @staticmethod
    def _build_source_cards(search_results: Sequence[Tuple[Document, float]]) -> List[Dict[str, str]]:
        """Format source information for the sidebar/UI."""
        source_cards: List[Dict[str, str]] = []
        seen = set()

        for rank, (document, distance) in enumerate(search_results, start=1):
            source_name = document.metadata.get("source", "Unknown")
            if source_name in seen:
                continue

            seen.add(source_name)
            source_cards.append(
                {
                    "rank": str(rank),
                    "source": source_name,
                    "score": f"{distance:.4f}",
                    "preview": document.page_content[:180].strip().replace("\n", " "),
                }
            )

        return source_cards

    @staticmethod
    def _build_chunk_cards(search_results: Sequence[Tuple[Document, float]]) -> List[Dict[str, str]]:
        """Format retrieved chunks for display and debugging."""
        chunk_cards: List[Dict[str, str]] = []

        for rank, (document, distance) in enumerate(search_results, start=1):
            chunk_cards.append(
                {
                    "rank": str(rank),
                    "source": document.metadata.get("source", "Unknown"),
                    "chunk_id": str(document.metadata.get("chunk_id", "-")),
                    "score": f"{distance:.4f}",
                    "content": document.page_content.strip(),
                }
            )

        return chunk_cards
