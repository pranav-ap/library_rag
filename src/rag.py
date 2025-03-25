from utils import logger
import os
from langgraph.graph import StateGraph, START, END
from langfuse.callback import CallbackHandler

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


class RAG:
    def __init__(self):
        self.llm, self.llm_json_mode = self.setup_llm()
        self.retriever = self.setup_retriever()
        self.web_search_tool = self.setup_web_search_tool()

        workflow = self.setup_workflow()
        self.graph = workflow.compile()

    @staticmethod
    def setup_llm():
        from langchain_ollama import ChatOllama
        local_llm = "llama3.2:3b-instruct-fp16"
        llm = ChatOllama(model=local_llm, temperature=0.2)
        llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

        return llm, llm_json_mode

    @staticmethod
    def setup_web_search_tool():
        from langchain_community.tools import TavilySearchResults
        web_search_tool = TavilySearchResults(max_results=3)
        return web_search_tool

    @staticmethod
    def setup_retriever():
        from src import DocumentIndexer
        retriever = DocumentIndexer()
        retriever.load_retriever()

        return retriever

    def setup_workflow(self):
        workflow = StateGraph(State)

        workflow.add_node(
            "reformulate_query",
            lambda state: reformulate_query(state, self.llm_json_mode)
        )

        workflow.set_entry_point("reformulate_query")

        workflow.add_conditional_edges(
            "reformulate_query",
            lambda state: route_question(state, self.llm_json_mode),
            {
                "web_search": "web_search",
                "vectorstore": "retrieve_documents",
            },
        )

        workflow.add_node(
            "web_search",
            lambda state: web_search(state, self.web_search_tool)
        )

        workflow.add_node(
            "retrieve_documents",
            lambda state: retrieve_documents(state, self.retriever)
        )

        workflow.add_node(
            "filter_relevant_documents",
            lambda state: filter_relevant_documents(state, self.llm_json_mode)
        )

        workflow.add_node(
            "generate_answer",
            lambda state: generate_answer(state, self.llm)
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
            lambda state: check_for_hallucinations(state, self.llm_json_mode),
            {
                "facts not useful": "web_search",
                "bad answer": "generate_answer",
                "max generation attempts reached": END,
                "done": END,
            },
        )

        return workflow


def setup_env_variables():
    os.environ["TAVILY_API_KEY"] = "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"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-38f44a84-ed8a-4998-9ce7-f1847ea68f9a"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-eb310539-6db0-4a7a-b4d5-66442841ded5"
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"


def make_rag():
    setup_env_variables()
    langfuse_handler = CallbackHandler()
    rag = RAG()
    return rag, langfuse_handler
