from langchain_community.vectorstores import FAISS
from tools.embed_model import CustomHFEmbeddings
from pathlib import Path

def load_db(index_dir: Path, model_name: str) -> FAISS:
    embed = CustomHFEmbeddings(model_name=model_name)
    db = FAISS.load_local(str(index_dir), embed, allow_dangerous_deserialization=True)
    return db

def query(db: FAISS, text: str, k: int = 5):
    return db.similarity_search(text, k=k)