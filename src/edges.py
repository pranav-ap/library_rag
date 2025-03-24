import json
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from .state import State
from .doc_utils import langchain_docs_to_string


def route_question(state: State, llm_json_mode):
    instructions = """
    You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.
    Return JSON with single key, datasource, that is 'web_search' or 'vectorstore' depending on the question.
    """

    question = state["question"]

    route_question = llm_json_mode.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=question)
    ])

    source = json.loads(route_question.content)["datasource"]

    if source == "web_search":
        return "web_search"

    return "vectorstore"


def is_extra_web_search_needed(state: State):
    must_web_search = state["must_web_search"]

    if must_web_search == "Yes":
        return "must_web_search"
    else:
        return "generate_answer"


def _grade_answer(llm_json_mode, instructions, prompt):
    result: AIMessage = llm_json_mode.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=prompt)
    ])

    return json.loads(result.content)["binary_score"]


def _check_relevance_of_answer_to_facts(llm_json_mode, documents, answer):
    instructions = """
    You are a teacher grading a quiz. 
    You will be given FACTS and a STUDENT ANSWER. 
    Here is the grade criteria to follow:
    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
    Score:
    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
    Avoid simply stating the correct answer at the outset.
    """

    prompt = """
    FACTS: \n\n {documents} \n\n STUDENT ANSWER: {answer}. 
    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score.
    """

    prompt_formatted = prompt.format(
        documents=langchain_docs_to_string(documents),
        answer=answer
    )

    fact_check_score = _grade_answer(llm_json_mode, instructions, prompt_formatted)

    return fact_check_score


def _check_relevance_of_answer_to_question(llm_json_mode, question, answer):
    instructions = """
    You are a teacher grading a quiz. 
    You will be given a QUESTION and a STUDENT ANSWER. 
    Here is the grade criteria to follow:
    (1) The STUDENT ANSWER helps to answer the QUESTION
    Score:
    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
    The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
    Avoid simply stating the correct answer at the outset.
    """

    prompt = """
    QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 
    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score.
    """

    prompt_formatted = prompt.format(
        question=question,
        generation=answer
    )

    relevance_score = _grade_answer(llm_json_mode, instructions, prompt_formatted)

    return relevance_score


def check_for_hallucinations(state: State, llm_json_mode):
    question, documents, answer = state["question"], state["documents"], state["answer"]
    max_retries, loop_step = state.get("MAX_ANSWER_GENERATIONS_RETRIES", 3), state["loop_step"]

    fact_check_score = _check_relevance_of_answer_to_facts(llm_json_mode, documents, answer)

    if fact_check_score == "no":
        if loop_step > max_retries:
            return "max generation attempts reached"

        return "facts not useful"

    relevance_score = _check_relevance_of_answer_to_question(llm_json_mode, question, answer)

    if relevance_score == "yes":
        return "done"

    if loop_step > max_retries:
        return "max generation attempts reached"

    return "bad answer"
