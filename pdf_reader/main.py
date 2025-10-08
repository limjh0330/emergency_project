from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from pdf_prep import pdf_to_docs
from rag_prep import chunk_docs, save_jsonl
from make_index import build_faiss

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="/pdf_reader/pdf/KTAS.pdf",
        help="입력 PDF")
    parser.add_argument(
        "--method",
        choices=["unstructuredpdfloader","marker", "pdfplumberloader","pymupdf", "pypdfium2", "opendataloader", "internvl", "dolphin"],
        default="pypdfium2",
    )
    args = parser.parse_args()
    project_root = str(Path(__file__).resolve().parents[1])
    start_page, end_page = 1,5

    pdf_path = project_root + args.file
    docs = pdf_to_docs(pdf_path, method=args.method)
    chunks = chunk_docs(docs, chunk_size=1000, chunk_overlap=150)
    jsonl_path = project_root+f"/pdf_reader/pdf_result/{args.method}_chunks.jsonl"
    save_jsonl(chunks, jsonl_path)
    print(f"[INFO] json 파일을 저장했습니다: {jsonl_path}")

    index_dir = project_root + "/pdf_result/index/ktas_faiss"
    model="sentence-transformers/all-MiniLM-L6-v2"
    build_faiss(jsonl_path = jsonl_path, index_dir = index_dir, embed_model = model)
    print(f"building FAISS for {jsonl_path}")

    embed = HuggingFaceEmbeddings(model_name=model)
    db = FAISS.load_local(index_dir, embed, allow_dangerous_deserialization=True)

    query = "KTAS 2단계 기준과 예시 증상은?"
    docs = db.similarity_search(query, k=5)
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.metadata} \n{d.page_content[:300]}...\n")