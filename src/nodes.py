import json
from langchain.schema import Document, HumanMessage, SystemMessage, AIMessage
from .state import State
from .doc_utils import langchain_docs_to_string


def reformulate_query(state, llm_json_mode):
    question = state["question"]

    system_prompt = """
    "You are an expert in information retrieval. 
    Your task is to improve user queries for better document retrieval. 
    You will rewrite the given query to make it more specific, informative, and 
    clear while preserving its original intent. Expand abbreviations, clarify ambiguity, 
    and add context if necessary. If the query is already optimal, return it unchanged.
    """

    user_prompt = """
    Return JSON with two two keys, new_question, which contains the improved query. 
    And a key, explanation, that contains an explanation for the changes made if any. State 'No changes' if none were needed.
    
    Query: {question}
    Answer:
    """

    user_prompt = user_prompt.format(
        question=question,
    )

    result: AIMessage = llm_json_mode.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    new_question = json.loads(result.content)['new_question']
    print(f"Reformulated question : {new_question}")

    return {"original_question": question, "question": new_question}


def retrieve_documents(state: State, retriever):
    question = state["question"]
    documents = retriever(question)

    print(f"Retrieved {len(documents)} documents.")

    return {"documents": documents}


def filter_relevant_documents(state: State, llm_json_mode):
    question: str = state["question"]
    documents: [Document] = state["documents"]

    instructions = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    """

    prompt = """
    Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question.
    """

    filtered_docs: [Document] = []
    must_web_search: bool = False

    for d in documents:
        prompt_formatted: str = prompt.format(document=d.page_content, question=question)

        result: AIMessage = llm_json_mode.invoke([
            SystemMessage(content=instructions),
            HumanMessage(content=prompt_formatted)
        ])

        grade = json.loads(result.content)["binary_score"]

        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            must_web_search = True

    return {"documents": filtered_docs, "must_web_search": must_web_search}


def web_search(state: State, web_search_tool):
    question = state["question"]
    documents = state.get("documents", [])

    docs = web_search_tool.invoke({"query": question})

    if len(docs) > 0 and isinstance(docs[0], dict):
        web_results: [Document] = [Document(page_content=d['content']) for d in docs]
        documents.extend(web_results)

    return {"documents": documents}


def generate_answer(state: State, llm):
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    MAX_ANSWER_LENGTH = state.get("MAX_ANSWER_LENGTH", 100)

    prompt = """
    You are an assistant for question-answering tasks. 
    Here is the context to use to answer the question:
    {context} 
    Think carefully about the above context. 
    Now, review the user question:
    {question}
    Provide an answer to this questions using only the above context. 
    Use a MAXIMUM of {MAX_ANSWER_LENGTH} words and keep the answer concise.
    Answer:
    """

    context = langchain_docs_to_string(documents)
    prompt_formatted = prompt.format(
        context=context,
        question=question,
        MAX_ANSWER_LENGTH=MAX_ANSWER_LENGTH,
    )

    result: AIMessage = llm.invoke([
        HumanMessage(content=prompt_formatted)
    ])

    return {"answer": result.content, "loop_step": loop_step + 1}

