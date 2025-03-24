import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document


class State(TypedDict):
    original_question: str
    question: str

    MAX_ANSWER_GENERATIONS_RETRIES: int
    MAX_ANSWER_LENGTH: int
    loop_step: int

    documents: List[Document]

    must_web_search: bool
    answer: str
