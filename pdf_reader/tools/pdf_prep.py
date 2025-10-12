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
    return "\n\n".join(doc for doc in docs)


# ---------- 4) LangChain: PyPDFium2Loader ----------
def pdf_to_text_pypdfium2(file_path: str) -> str:
    """
    LangChain PyPDFium2Loader로 텍스트 추출
    pip install -U langchain-community pypdfium2
    """
    from langchain_community.document_loaders import PyPDFium2Loader  # type: ignore
    loader = PyPDFium2Loader(file_path, extract_images=False)
    docs = loader.load()
    return docs

# ---------- 5) ByteDance Dolphin ----------
def pdf_to_text_dolphin(
    file_path: str,
    model_dir: str = "./pdf_reader/refer_model/hf_model",  # huggingface-cli로 받은 ByteDance/Dolphin 로컬 경로
    save_dir: Optional[str] = None,
    max_batch_size: int = 8,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None
) -> str:
    """
    Dolphin 페이지 파싱(HF 프레임워크) 스크립트를 서브프로세스로 호출.
    결과 폴더의 Markdown/JSON을 읽어 텍스트로 리턴.

    설치 요약:
      conda install -c conda-forge git-lfs
      git lfs install
      git clone https://huggingface.co/ByteDance/Dolphin ./hf_model
      pip install -r https://raw.githubusercontent.com/bytedance/Dolphin/master/requirements.txt
    """
    import torch, tempfile, os, subprocess
    torch.set_default_dtype(torch.float32)
    
    # 결과 저장 경로
    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="dolphin_out_")

    candidate_paths = [
        "./pdf_reader/refer_model/Dolphin/demo_page_hf.py",
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
    "python", demo_script,
    "--model_path", model_dir,
    "--input_path", file_path,
    "--save_dir", save_dir,
    "--max_batch_size", str(max_batch_size),
    ]
    if start_page is not None:
        cmd += ["--start_page", str(start_page)]
    if end_page is not None:
        cmd += ["--end_page", str(end_page)]
    
    cmd = list(map(str, cmd))
    env = os.environ.copy()
    env["ACCELERATE_DISABLE_MIXED_PRECISION"] = "1"

    subprocess.run(cmd, check=True, env=env)

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
# ---------- 6) Marker ----------
# def pdf_to_text_marker(file_path: str, output_format: str = "markdown") -> str:
#     """
#     marker 라이브러리를 활용하여 PDF → 텍스트(또는 Markdown) 변환.
#     marker는 PDF 레이아웃과 구조를 유지하며 텍스트를 추출할 수 있음.

#     설치:
#       pip install -U marker-pdf  # (공식: https://github.com/datalab-to/marker)

#     매개변수:
#       file_path : PDF 파일 경로
#       output_format : 'markdown' | 'html' | 'text' | 'json'

#     반환:
#       변환된 텍스트 문자열
#     """
#     import os
#     from marker.convert import convert_single_pdf

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")

#     # 변환 수행 (marker는 내부적으로 PyMuPDF + ML 기반 PDF 구조 파서 사용)
#     result = convert_single_pdf(file_path, format=output_format)

#     # convert_single_pdf()는 변환 결과를 문자열 형태로 반환
#     if isinstance(result, dict):
#         # 일부 포맷(json 등)은 dict 형태로 반환됨 → JSON 문자열로 직렬화
#         import json
#         return json.dumps(result, ensure_ascii=False, indent=2)
#     else:
#         return str(result)

# ---------- 공통 진입점 ----------
def pdf_to_docs(file_path: str, start_page = 1, end_page = 1000, method = "pypdfium2") -> str:
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
        return pdf_to_text_dolphin(file_path, start_page=start_page, end_page=end_page)
    # if method == "marker":
    #     return pdf_to_text_marker(file_path)
    raise ValueError(f"Unknown method: {method}")