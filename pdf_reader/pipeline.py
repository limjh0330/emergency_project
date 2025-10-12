from pathlib import Path
from path_setting import Settings

from tools.pdf_prep import pdf_to_docs
from tools.rag_prep import chunk_docs, save_jsonl
from tools.make_index import build_faiss

def prepare_chunks(pdf_path: Path, method: str, chunk_size:int, chunk_overlap:int, cfg: Settings) -> Path:
    """PDF → 문서 → 청크 → JSONL 저장. 결과 JSONL 경로 반환."""
    docs = pdf_to_docs(str(pdf_path), method)
    chunks = chunk_docs(docs, chunk_size, chunk_overlap)

    out = cfg.jsonl_path(method)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(chunks, str(out))
    print(f"[INFO] 청크 JSONL 저장: {out}")
    return out

def build_index(jsonl_path: Path, llm_model:str, cfg: Settings) -> Path:
    """JSONL → FAISS 인덱스 구축. 인덱스 디렉터리 경로 반환."""
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    build_faiss(jsonl_path=str(jsonl_path),
                index_dir=str(cfg.index_dir),
                embed_model=llm_model)
    print(f"[INFO] FAISS 인덱스 구축 완료: {cfg.index_dir}")
    return cfg.index_dir