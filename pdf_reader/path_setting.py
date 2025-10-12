from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parents[1]   # 프로젝트 루트
    pdf_rel: str = "pdf_reader/pdf/KTAS.pdf"
    result_dir_rel: str = "pdf_reader/pdf_result"
    index_dir_rel: str = "pdf_reader/pdf_result/index/ktas_faiss"

    @property
    def pdf_path(self) -> Path:
        return self.base_dir / self.pdf_rel

    def jsonl_path(self, method: str) -> Path:
        return self.base_dir / self.result_dir_rel / f"{method}_chunks.jsonl"

    def txt_path(self, method: str) -> Path:
        return self.base_dir / self.result_dir_rel / f"{method}_raw.txt"
    
    @property
    def index_dir(self) -> Path:
        return self.base_dir / self.index_dir_rel