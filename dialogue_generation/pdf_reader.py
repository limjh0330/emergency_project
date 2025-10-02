from __future__ import annotations
import os
import io
import json
import tempfile
import subprocess
from typing import List, Literal, Optional

# ---------- 1) LangChain: UnstructuredPDFLoader ----------
def pdf_to_text_unstructuredpdfloader(file_path: str) -> str:
    """
    LangChain UnstructuredPDFLoader로 텍스트 추출
    pip install -U langchain-community pdfplumber
    """
    from langchain_community.document_loaders import UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(
        file_path,
        mode="elements",
        strategy="hi_res",
        infer_table_structure=True,
        languages=["kor", "eng"]
        # pages=[70]
    )
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

# ---------- 2) LangChain: PyMuPDFLoader ----------
def pdf_to_text_pdfplumberloader(file_path: str) -> str:
    """
    LangChain PyMuPDFLoader로 텍스트 추출
    pip install -U langchain-community pdfplumber
    """
    from langchain_community.document_loaders import PDFPlumberLoader
    loader = PDFPlumberLoader(file_path, extract_images = False) # pip install rapidocr-onnxruntime
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

# from langchain_core.documents import Document
# from unstructured.partition.pdf import partition_pdf
# import pdfplumber
# import pandas as pd

# elif loader_type == 'partition_pdf':
#     print(f'Use {loader_type}')
#     elements = partition_pdf(
#         filename=file_path,
#         strategy="hi_res",
#         infer_table_structure=True,
#         languages=["kor", "eng"],
#         # extract_images_in_pdf=False,
#     )

#     docs = []
#     for el in elements:
#         if len(docs)>=10:
#             break
#         # 우선순위: element.text → element.metadata.text_as_html → 빈 문자열
#         text = getattr(el, "text", None) or getattr(getattr(el, "metadata", None), "text_as_html", None) or ""
#         if not text:  # 완전 빈 요소는 스킵(원하면 남겨도 됨)
#             continue

#         meta = {}
#         if hasattr(el, "metadata") and el.metadata is not None:
#             # dict로 안전 변환
#             try:
#                 meta = el.metadata.to_dict()
#             except Exception:
#                 meta = dict(el.metadata.__dict__)

#         docs.append(Document(page_content=text, metadata=meta))


# ---------- 3) LangChain: PyMuPDFLoader ----------
def pdf_to_text_pymupdf(file_path: str) -> str:
    """
    LangChain PyMuPDFLoader로 텍스트 추출
    pip install -U langchain-community pymupdf
    """
    from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore
    # conda install -c conda-forge poppler
    # conda install -c conda-forge poppler tesseract

    loader = PyMuPDFLoader(file_path, extract_images=False)
    docs = loader.load()  # 페이지별 Document 리스트
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)


# ---------- 4) LangChain: PyPDFium2Loader ----------
def pdf_to_text_pypdfium2(file_path: str) -> str:
    """
    LangChain PyPDFium2Loader로 텍스트 추출
    pip install -U langchain-community pypdfium2
    """
    from langchain_community.document_loaders import PyPDFium2Loader  # type: ignore
    loader = PyPDFium2Loader(file_path, extract_images=False)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

# ---------- 5) ByteDance Dolphin ----------
def pdf_to_text_dolphin(
    file_path: str,
    model_dir: str = "./hf_model",  # huggingface-cli로 받은 ByteDance/Dolphin 로컬 경로
    save_dir: Optional[str] = None,
    max_batch_size: int = 8,
) -> str:
    """
    Dolphin 페이지 파싱(HF 프레임워크) 스크립트를 서브프로세스로 호출.
    결과 폴더의 Markdown/JSON을 읽어 텍스트로 리턴.

    설치 요약:
      git lfs install
      git clone https://huggingface.co/ByteDance/Dolphin ./hf_model
      pip install -r https://raw.githubusercontent.com/bytedance/Dolphin/master/requirements.txt
      # 또는 레포 클론 후 requirements 설치
      # 사용 예: demo_page_hf.py --model_path ./hf_model --input_path <pdf> --save_dir <out>
    레포 사용법 근거: README의 page-level parsing (HF) 섹션.  [oai_citation:2‡GitHub](https://github.com/bytedance/Dolphin)
    """
    # 결과 저장 경로
    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="dolphin_out_")

    # 레포의 demo_page_hf.py 사용 (로컬에 clone되어 있다고 가정하거나, 원격 URL로 임시 다운로드)
    # 가장 간단히: pip로 설치된 것이 아니라면, python -m pip -q로 다운로드/실행 로직을 추가해도 됨.
    # 여기서는 사용자가 레포를 클론했고, 루트에 demo_page_hf.py가 있다고 가정.
    # 경로 자동 탐색: 현재 작업 디렉토리에서 Dolphin 레포가 있는지 확인
    candidate_paths = [
        "./Dolphin/demo_page_hf.py",
        "./demo_page_hf.py",
    ]
    demo_script = None
    for p in candidate_paths:
        if os.path.exists(p):
            demo_script = p
            break
    if demo_script is None:
        raise FileNotFoundError(
            "demo_page_hf.py를 찾을 수 없습니다. bytedance/Dolphin 레포를 클론하고 스크립트 경로를 맞춰주세요."
        )

    # 실행
    cmd = [
        "python",
        demo_script,
        "--model_path",
        model_dir,
        "--input_path",
        file_path,
        "--save_dir",
        save_dir,
        "--max_batch_size",
        str(max_batch_size),
    ]
    subprocess.run(cmd, check=True)

    # 결과 폴더에서 Markdown 우선, 없으면 JSON, 없으면 TXT 읽기
    # demo_page_hf.py는 보통 페이지별 JSON/MD를 생성 (레포 가이드 기준)
    md_chunks, json_chunks, txt_chunks = [], [], []
    for root, _, files in os.walk(save_dir):
        for name in sorted(files):
            p = os.path.join(root, name)
            if name.lower().endswith((".md", ".markdown")):
                with open(p, "r", encoding="utf-8") as f:
                    md_chunks.append(f.read())
            elif name.lower().endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    json_chunks.append(f.read())
            elif name.lower().endswith(".txt"):
                with open(p, "r", encoding="utf-8") as f:
                    txt_chunks.append(f.read())

    if md_chunks:
        return "\n\n".join(md_chunks)
    if json_chunks:
        return "\n\n".join(json_chunks)
    if txt_chunks:
        return "\n\n".join(txt_chunks)

    # 아무 것도 없으면 빈 문자열
    return ""


# ---------- 공통 진입점 ----------
Method = Literal["pymupdf", "pypdfium2", "opendataloader", "internvl", "dolphin"]

def pdf_to_text(file_path: str, method: Method = "pymupdf") -> str:
    if method == "unstructuredpdfloader":
        return pdf_to_text_unstructuredpdfloader(file_path)
    if method == "pdfplumberloader":
        return pdf_to_text_pdfplumberloader(file_path)
    if method == "pymupdf":
        return pdf_to_text_pymupdf(file_path)
    if method == "pypdfium2":
        return pdf_to_text_pypdfium2(file_path)
    # if method == "opendataloader":
    #     return pdf_to_text_opendataloader_pdf(file_path)
    # if method == "internvl":
    #     return pdf_to_text_internvl(file_path)
    if method == "dolphin":
        return pdf_to_text_dolphin(file_path)
    raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="/Users/lim/Desktop/Desktop/study/Master/Emergency project/dialogue_generation/pdf/KTAS.pdf",
        help="입력 PDF 경로")
    parser.add_argument(
        "--method",
        choices=["unstructuredpdfloader","pdfplumberloader","pymupdf", "pypdfium2", "opendataloader", "internvl", "dolphin"],
        default="dolphin",
    )
    parser.add_argument(
        "--out",
        default="/Users/lim/Desktop/Desktop/study/Master/Emergency project/dialogue_generation/pdf_result",
        help="추출된 텍스트 저장 경로(.txt). 미지정 시 입력 PDF와 같은 경로에 .txt로 저장"
    )
    args = parser.parse_args()
    out_path = args.out+f"/{args.method}.txt"

    text = pdf_to_text(args.path, method=args.method)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")

    print(f"[INFO] 텍스트를 저장했습니다: {out_path}")
