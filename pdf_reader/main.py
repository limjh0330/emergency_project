import argparse

from path_setting import Settings
from pipeline import prepare_chunks, build_index
from tools.faiss_io import load_db, query

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default=None,  # None이면 Settings의 기본값 사용
        help="입력 PDF 경로 (프로젝트 루트 기준 상대경로 또는 절대경로)"
    )
    parser.add_argument(
        "--method",
        choices=[
            "unstructuredpdfloader","marker","pypdfium2",
            "pymupdf","pdfplumberloader","opendataloader","internvl","dolphin"
        ],
        default="marker",
        help="PDF 로딩 방식"
    )
    parser.add_argument(
        "--llm_model",
        choices=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "Qwen/Qwen3-Embedding-0.6B"
        ],
        default="Qwen/Qwen3-Embedding-0.6B",
        help="LLM 모델 선택"
    )
    parser.add_argument(
        "--chunk_size",
        default=1000
    )
    parser.add_argument(
        "--chunk_overlap",
        default=150
    )
    parser.add_argument(
        "--query",
        default="KTAS 2단계 기준과 예시 증상은?",
        help="테스트 질의"
    )
    parser.add_argument("--k", type=int, default=5, help="검색 상위 문서 수")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Settings()

    # 입력 PDF 경로 결정
    pdf_path = cfg.pdf_path
    assert pdf_path.exists(), f"PDF를 찾을 수 없습니다: {pdf_path}"

    # 1) 청크 준비 → JSONL 저장
    jsonl_path = prepare_chunks(pdf_path, args.method, args.chunk_size, args.chunk_overlap, cfg)

    # 2) 인덱스 구축
    index_dir = build_index(jsonl_path, args.llm_model, cfg)

    # 3) 인덱스 로드 & 질의
    db = load_db(index_dir, args.llm_model)
    docs = query(db, args.query, k=args.k)

    for i, d in enumerate(docs, 1):
        preview = d.page_content[:300].replace("\n", " ")
        print(f"[{i}] {d.metadata}\n{preview}...\n")

if __name__ == "__main__":
    main()