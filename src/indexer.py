import os
import pickle
import shutil
import numpy as np
from collections import defaultdict
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader
)
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from .doc_utils import replace_t_with_space

CHROMA_DB_PATH = "D:/code/library_rag/storage/"
BM25_CACHE_PATH = "D:/code/library_rag/storage/bm25.pkl"


class DocumentIndexer:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.bm25_retriever = None

    @staticmethod
    def load_documents(paths: List[str]) -> List[Document]:
        documents = []

        for path in paths:
            if path.startswith("http://") or path.startswith("https://"):
                loader = WebBaseLoader(path)
            else:
                # Handle local files
                ext = path.lower().split(".")[-1]

                if ext == "pdf":
                    loader = PyPDFLoader(path)
                elif ext == "txt":
                    loader = TextLoader(path)
                elif ext == "md":
                    loader = UnstructuredMarkdownLoader(path)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")

            docs = loader.load()
            documents.extend(docs)

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return text_splitter.split_documents(documents)

    def index_documents(self, file_paths: List[str]):
        # clear directory
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)

        documents = self.load_documents(file_paths)
        doc_chunks = self.split_documents(documents)
        doc_chunks = replace_t_with_space(doc_chunks)

        self.vectorstore = Chroma.from_documents(
            documents=doc_chunks,
            collection_name="rag_chroma",
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_PATH,
        )

        self.bm25_retriever = BM25Retriever.from_documents(doc_chunks)
        self.bm25_retriever.k = 3

        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump(self.bm25_retriever, f)

        print(f"Indexed {len(doc_chunks)} chunks from {len(file_paths)} files.")

    def load_retriever(self, k=3):
        if os.path.exists(CHROMA_DB_PATH):
            print(f'Loading Chroma from {CHROMA_DB_PATH}')
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings
            ).as_retriever(k=k)

        if os.path.exists(BM25_CACHE_PATH):
            print(f'Loading BM25 from {BM25_CACHE_PATH}')
            with open(BM25_CACHE_PATH, "rb") as f:
                self.bm25_retriever = pickle.load(f)

    def fusion_retrieval(self, query: str, k: int = 3, alpha: float = 0.5) -> List[Document]:
        assert self.vectorstore is not None, "Vectorstore is not loaded"
        assert self.bm25_retriever is not None, "BM25 retriever is not loaded"

        vector_results = self.vectorstore.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)

        # Assign initial scores

        vector_scores = {
            doc.page_content: score for doc, score in zip(
                vector_results,
                np.linspace(1, 0, len(vector_results))
            )
        }

        bm25_scores = {
            doc.page_content: score for doc, score in zip(
                bm25_results,
                np.linspace(1, 0, len(bm25_results))
            )
        }

        # Normalize scores

        combined_scores = defaultdict(float)

        for doc_content in set(vector_scores.keys()).union(bm25_scores.keys()):
            v_score = vector_scores.get(doc_content, 0)
            b_score = bm25_scores.get(doc_content, 0)
            combined_scores[doc_content] = alpha * v_score + (1 - alpha) * b_score

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Retrieve actual document objects in ranked order
        doc_lookup = {doc.page_content: doc for doc in vector_results + bm25_results}

        k = min(k, len(sorted_docs))

        return [doc_lookup[doc_content] for doc_content, _ in sorted_docs[:k]]
