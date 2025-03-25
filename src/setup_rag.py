from utils import logger
import os
from langgraph.graph import StateGraph, START, END

from .state import State

from .nodes import (
    reformulate_query,
    retrieve_documents,
    generate_answer,
    filter_relevant_documents,
    web_search
)

from .edges import (
    route_question,
    is_extra_web_search_needed,
    check_for_hallucinations
)


def setup_env_variables():
    os.environ["TAVILY_API_KEY"] = "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"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-38f44a84-ed8a-4998-9ce7-f1847ea68f9a"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-eb310539-6db0-4a7a-b4d5-66442841ded5"
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"


def setup_llm():
    from langchain_ollama import ChatOllama
    local_llm = "llama3.2:3b-instruct-fp16"
    llm = ChatOllama(model=local_llm, temperature=0.2)
    llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

    return llm, llm_json_mode


def setup_web_search_tool():
    from langchain_community.tools import TavilySearchResults
    web_search_tool = TavilySearchResults(max_results=3)
    return web_search_tool


def setup_retriever():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.retrievers import BM25Retriever

    urls = ["https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://www.anthropic.com/engineering/building-effective-agents"]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    doc_splits = text_splitter.split_documents(docs_list)

    CHROMA_DB_PATH = "./../storage"

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )

    vector_retriever = vectorstore.as_retriever(k=3)

    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    bm25_retriever.k = 3

    def hybrid_retriever(query):
        bm25_results = bm25_retriever.invoke(query)
        vector_results = vector_retriever.invoke(query)
        return bm25_results + vector_results

    return hybrid_retriever


def setup_workflow():
    llm, llm_json_mode = setup_llm()
    retriever = setup_retriever()
    web_search_tool = setup_web_search_tool()

    workflow = StateGraph(State)

    workflow.add_node(
        "reformulate_query",
        lambda state: reformulate_query(state, llm_json_mode)
    )

    workflow.set_entry_point("reformulate_query")

    workflow.add_conditional_edges(
        "reformulate_query",
        lambda state: route_question(state, llm_json_mode),
        {
            "web_search": "web_search",
            "vectorstore": "retrieve_documents",
        },
    )

    workflow.add_node(
        "web_search",
        lambda state: web_search(state, web_search_tool)
    )

    workflow.add_node(
        "retrieve_documents",
        lambda state: retrieve_documents(state, retriever)
    )

    workflow.add_node(
        "filter_relevant_documents",
        lambda state: filter_relevant_documents(state, llm_json_mode)
    )

    workflow.add_node(
        "generate_answer",
        lambda state: generate_answer(state, llm)
    )

    workflow.add_edge("retrieve_documents", "filter_relevant_documents")

    workflow.add_conditional_edges(
        "filter_relevant_documents",
        is_extra_web_search_needed,
        {
            "must_web_search": "web_search",
            "generate_answer": "generate_answer",
        },
    )

    workflow.add_edge("web_search", "generate_answer")

    workflow.add_conditional_edges(
        "generate_answer",
        lambda state: check_for_hallucinations(state, llm_json_mode),
        {
            "facts not useful": "web_search",
            "bad answer": "generate_answer",
            "max generation attempts reached": END,
            "done": END,
        },
    )

    return workflow
