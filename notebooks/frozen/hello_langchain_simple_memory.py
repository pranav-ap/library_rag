from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


system_prompt = "You are a helpful assistant called Zeta."

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])

chat_map = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()

    return chat_map[session_id]


def main():
    model_name = "qwen2.5:7b"
    llm = ChatOllama(temperature=0.0, model=model_name)

    pipeline = prompt_template | llm

    pipeline_with_history = RunnableWithMessageHistory(
        pipeline,
        get_session_history=get_chat_history,
        input_messages_key="query",
        history_messages_key="history"
    )

    res = pipeline_with_history.invoke(
        input={"query": "Hi, my name is Josh"},
        config={"session_id": "id_123"},
    )

    print(res.content)


if __name__ == "__main__":
    main()
