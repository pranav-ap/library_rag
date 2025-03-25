import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document


CHROMA_DB_PATH = "./storage"


def load_documents(urls=None, file_paths=None):
    docs = []

    if urls:
        web_docs = [WebBaseLoader(url).load() for url in urls]
        docs.extend([item for sublist in web_docs for item in sublist])

    if file_paths:
        file_docs = [TextLoader(path).load() for path in file_paths]
        docs.extend([item for sublist in file_docs for item in sublist])

    return docs


def preprocess_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )

    return text_splitter.split_documents(docs)


def store_documents(doc_splits):
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag_chroma",
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=CHROMA_DB_PATH
    )

    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    bm25_retriever.k = 3

    return vectorstore.as_retriever(k=3), bm25_retriever


def hybrid_retriever(query, vector_retriever, bm25_retriever):
    bm25_results = {doc.page_content: 1 for doc in bm25_retriever.get_relevant_documents(query)}
    vector_results = {doc.page_content: 1 for doc in vector_retriever.get_relevant_documents(query)}

    combined_results = list(set(bm25_results.keys()) | set(vector_results.keys()))
    return [Document(page_content=text) for text in combined_results]


def main():
    urls = ["https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://www.anthropic.com/engineering/building-effective-agents"]

    file_paths = []

    docs = load_documents(urls, file_paths)
    doc_splits = preprocess_documents(docs)
    _, _ = store_documents(doc_splits)
    print('done!')


if __name__ == "__main__":
    main()
