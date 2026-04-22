"""Document loading, cleaning, deduplication, and chunking."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig


class DocumentProcessor:
    """Handle uploaded files and convert them into clean LangChain documents."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""],
        )

    def save_uploaded_files(self, uploaded_files: Sequence) -> List[Path]:
        """Persist Streamlit uploaded files to the local upload directory."""
        saved_paths: List[Path] = []

        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix not in self.config.allowed_extensions:
                continue

            safe_stem = self._sanitize_filename(Path(uploaded_file.name).stem)
            target_path = self.config.upload_dir / f"{safe_stem}{suffix}"
            duplicate_index = 1

            while target_path.exists():
                target_path = self.config.upload_dir / f"{safe_stem}_{duplicate_index}{suffix}"
                duplicate_index += 1

            target_path.write_bytes(uploaded_file.getbuffer())
            saved_paths.append(target_path)

        return saved_paths

    def load_all_documents(self) -> List[Document]:
        """Read every saved file from the upload directory."""
        paths = sorted(
            path
            for path in self.config.upload_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.config.allowed_extensions
        )
        return self.load_documents_from_paths(paths)

    def load_documents_from_paths(self, paths: Iterable[Path]) -> List[Document]:
        """Load PDF/TXT files and build LangChain Document objects."""
        documents: List[Document] = []

        for path in paths:
            raw_text = self._read_file(path)
            cleaned_text = self.clean_text(raw_text)
            if not cleaned_text:
                continue

            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                    },
                )
            )

        return documents

    def split_and_deduplicate(self, documents: Sequence[Document]) -> List[Document]:
        """Split long documents into chunks and remove duplicate chunks."""
        split_docs = self.text_splitter.split_documents(list(documents))
        unique_chunks: List[Document] = []
        seen_hashes = set()

        for chunk in split_docs:
            normalized_text = self._normalize_for_hash(chunk.page_content)
            if len(normalized_text) < self.config.min_chunk_length:
                continue

            content_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)
            chunk.metadata["chunk_id"] = len(unique_chunks) + 1
            chunk.metadata["char_count"] = len(chunk.page_content)
            unique_chunks.append(chunk)

        return unique_chunks

    def clear_uploaded_files(self) -> None:
        """Delete saved source files without touching project code."""
        for path in self.config.upload_dir.iterdir():
            if path.is_file():
                path.unlink()

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and remove obvious duplicated lines."""
        if not text:
            return ""

        text = text.replace("\x00", " ").replace("\u3000", " ").replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)

        cleaned_lines = []
        previous_line = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                previous_line = None
                continue

            if line == previous_line:
                continue

            cleaned_lines.append(line)
            previous_line = line

        normalized = "\n".join(cleaned_lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Keep filenames predictable for local storage."""
        sanitized = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "_", name).strip("._")
        return sanitized or "document"

    @staticmethod
    def _normalize_for_hash(text: str) -> str:
        """Remove formatting differences before deduplication."""
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _read_file(path: Path) -> str:
        """Dispatch file loading based on extension."""
        if path.suffix.lower() == ".pdf":
            return DocumentProcessor._read_pdf(path)
        return DocumentProcessor._read_txt(path)

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract raw text from each PDF page."""
        reader = PdfReader(str(path))
        pages: List[str] = []

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                pages.append(f"[Page {page_index}]\n{page_text}")

        return "\n\n".join(pages)

    @staticmethod
    def _read_txt(path: Path) -> str:
        """Read text files with a small set of common encodings."""
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")
