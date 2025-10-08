from __future__ import annotations
import json
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_chunks_jsonl(path: str) -> List[Document]:
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec: Dict[str, Any] = json.loads(line)
            docs.append(Document(page_content=rec["text"], metadata=rec.get("metadata") or {}))
    return docs

def build_faiss(jsonl_path: str, index_dir: str, embed_model : str) -> None:
    docs = load_chunks_jsonl(jsonl_path)
    embed = HuggingFaceEmbeddings(model_name=embed_model)  # pip install sentence-transformers
    db = FAISS.from_documents(docs, embed)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    db.save_local(index_dir)
    print(f"[INFO] Built FAISS index at: {index_dir}")