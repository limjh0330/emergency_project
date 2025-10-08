from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(s: str) -> str:
    # PDF에서 흔한 잡스페이스/헤더·푸터 패턴 간단 정리(필요 시 커스텀)
    s = s.replace("\x00", "").strip()
    s = re.sub(r"[ \t]+\n", "\n", s)      # 줄끝 공백 제거
    s = re.sub(r"\n{3,}", "\n\n", s)      # 과도한 개행 압축
    return s

def chunk_docs(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # 자연스러운 경계 우선
    )
    # 메타데이터 유지한 채로 분할
    return splitter.split_documents([
        Document(page_content=clean_text(d.page_content or ""),
                 metadata=d.metadata or {})
        for d in docs
        if getattr(d, "page_content", "")
    ])

def save_jsonl(chunks: List[Document], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, d in enumerate(chunks, 1):
            rec: Dict[str, Any] = {
                "id": i,
                "text": d.page_content,
                "metadata": d.metadata,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def docs_to_chunk(docs, path: str) -> None:
    chunks = chunk_docs(docs, chunk_size=1000, chunk_overlap=150)
    save_jsonl(chunks, path)

