from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain_community.vectorstores import DocArrayInMemorySearch


def main():
    model_name = "qwen2.5:7b"
    llm = ChatOllama(temperature=0.0, model=model_name)
    embedding = OllamaEmbeddings(model="nomic-embed-text")

    vecstore_a = DocArrayInMemorySearch.from_texts(
        ["James' birthday is December the 7th"],
        embedding=embedding
    )
    vecstore_b = DocArrayInMemorySearch.from_texts(
        ["James was born in the year 1994"],
        embedding=embedding
    )

    prompt_str = """Using the context provided, answer the user's question.
    Context: 
    {context_a}
    {context_b}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt_str),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    from langchain_core.runnables import RunnablePassthrough, RunnableParallel

    retriever_a = vecstore_a.as_retriever()
    retriever_b = vecstore_b.as_retriever()

    retrieval = RunnableParallel(
        {
            "context_a": retriever_a,
            "context_b": retriever_b,
            "question": RunnablePassthrough()
        }
    )

    from langchain.schema.output_parser import StrOutputParser
    output_parser = StrOutputParser()

    chain = retrieval | prompt | llm | output_parser

    result = chain.invoke("What was the date when James was born")
    print(result)


if __name__ == "__main__":
    main()
