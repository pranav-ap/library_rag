import os
import pickle
import re
import string

import chromadb
from sklearn.feature_extraction import _stop_words
from rank_bm25 import BM25Okapi

from .embedding import SnippetEmbeddingFunction

from utils import logger, make_clear_directory
from config import config


def bm25_tokenizer(text):
    tokenized_doc = []

    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)

    return tokenized_doc


class DataStore:
    def __init__(self, persistent=True, reset=False):
        if persistent:
            logger.info("Using persistent storage")

        if reset:
            logger.info("Resetting storage directory")
            make_clear_directory(config.paths.storage)

        self.chroma_client = chromadb.PersistentClient(path=config.paths.storage) if persistent else chromadb.EphemeralClient()

        self.chroma_collection = None
        self.bm25_collection = None

    @staticmethod
    def _get_documents():
        from utils import list_files_in_folder
        paths = list_files_in_folder(config.paths.roots.data)

        logger.info('Reading Documents')

        from llama_index.core import SimpleDirectoryReader
        documents = SimpleDirectoryReader(
            input_dir=config.paths.roots.data,
        ).load_data()

        # from llama_index.core import Document
        # documents: [Document] = []
        #
        # for file_path in paths:
        #     with open(file_path, "r", encoding="utf-8") as file:
        #         text = file.read()
        #
        #     text = re.sub('\r', ' ', text)
        #     text = re.sub('\\s+', ' ', text)
        #     text = text.strip()
        #
        #     file_name = os.path.basename(file_path)
        #     doc = Document(
        #         doc_id=file_name,
        #         text=text,
        #         extra_info={
        #             'file_path': file_path,
        #             'file_name': file_name,
        #             'file_type': 'text/plain',
        #         }
        #     )
        #
        #     documents.append(doc)

        return documents

    def _get_snippet_nodes(self):
        documents = self._get_documents()

        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter

        logger.info('Ingesting Documents')

        pipeline = IngestionPipeline(transformations=[
                SentenceSplitter(
                    chunk_size=config.task.chunk_size,
                    chunk_overlap=50,
                )
            ])

        nodes = pipeline.run(documents=documents)

        return nodes

    def _setup_chromadb(self):
        nodes = self._get_snippet_nodes()

        logger.info('Setup ChromaDB')

        collection = self.chroma_client.get_or_create_collection(
            name=config.task.collection_name,
            embedding_function=SnippetEmbeddingFunction(),
            metadata={
                "description": "Store vectors for RAG",
            }
        )

        # noinspection SpellCheckingInspection
        ids, documents, metadatas = [], [], []

        for node in nodes:
            ids.append(node.node_id)
            documents.append(node.text)
            metadatas.append({
                'file_name': node.metadata.get('file_name', 'unknown'),
                'file_type': node.metadata.get('file_type', 'unknown'),
            })

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return collection

    @staticmethod
    def _setup_bm25(collection):
        logger.info('Setup BM25')

        result = collection.get()

        ids = result['ids']
        documents = result['documents']

        tokenized_documents = [bm25_tokenizer(doc) for doc in documents]
        index = BM25Okapi(tokenized_documents)

        collection = {
            "index": index,
            "ids": ids,
            "documents": documents,
        }

        collections = {
            config.task.collection_name: collection
        }

        with open(f"{config.paths.storage}/bm25.pkl", "wb") as f:
            pickle.dump(collections, f)

        return collection

    def _load_chroma_collection(self):
        collection = self.chroma_client.get_or_create_collection(
            name=config.task.collection_name,
            embedding_function=SnippetEmbeddingFunction()
        )

        return collection

    @staticmethod
    def _load_bm25_collection():
        with open(f"{config.paths.storage}/bm25.pkl", "rb") as f:
            collections = pickle.load(f)
            collection = collections[config.task.collection_name]

        return collection

    def populate(self):
        chroma_collection = self._setup_chromadb()
        bm25_collection = self._setup_bm25(chroma_collection)

        logger.info("Done!")

        return chroma_collection, bm25_collection

    def load(self):
        self.chroma_collection = self._load_chroma_collection()
        self.bm25_collection = self._load_bm25_collection()
