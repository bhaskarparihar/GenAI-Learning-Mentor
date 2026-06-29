# ============================================================
# rag_engine.py - RAG Pipeline with ChromaDB & LangChain
# ============================================================

import os
import shutil
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Embedding model selection ────────────────────────────────
def get_embeddings(provider: str = "gemini"):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


CHROMA_DIR = "./chroma_db"


class RAGEngine:
    """Handles document ingestion, chunking, embedding and retrieval."""

    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        self.embeddings = get_embeddings(provider)
        self.vectorstore: Optional[Chroma] = None
        self._load_existing_store()

    # ── Internal helpers ─────────────────────────────────────
    def _load_existing_store(self):
        if os.path.exists(CHROMA_DIR):
            try:
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=self.embeddings,
                )
            except Exception:
                self.vectorstore = None

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(docs)

    # ── Public API ───────────────────────────────────────────
    def ingest_pdf(self, file_path: str) -> int:
        """Load a PDF, chunk it, and add to the vector store. Returns chunk count."""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = self._chunk_documents(docs)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=CHROMA_DIR,
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.persist()
        return len(chunks)

    def ingest_text(self, text: str, source: str = "manual_input") -> int:
        """Ingest raw text directly into the vector store."""
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self._chunk_documents([doc])

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=CHROMA_DIR,
            )
        else:
            self.vectorstore.add_documents(chunks)

        self.vectorstore.persist()
        return len(chunks)

    def get_retriever(self, k: int = 5):
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )

    def clear_store(self):
        """Wipe all stored vectors (reset knowledge base)."""
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        self.vectorstore = None

    def has_documents(self) -> bool:
        if self.vectorstore is None:
            return False
        try:
            return self.vectorstore._collection.count() > 0
        except Exception:
            return False
