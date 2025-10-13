from langchain_core.embeddings import Embeddings
import torch
from transformers import AutoTokenizer, AutoModel

class CustomHFEmbeddings(Embeddings):
    def __init__(self, model_name="bert-base-uncased", pool="mean"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.pool = pool

    def embed_query(self, text: str):
        return self._embed([text])[0]

    def embed_documents(self, texts: list[str]):
        return self._embed(texts)

    def _embed(self, texts: list[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.pool == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()