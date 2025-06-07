from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)

from langchain_core.tools import tool
import json


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y


@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x


def main():
    model_name = "qwen2.5:7b"
    llm = ChatOllama(temperature=0.0, model=model_name)

    print(add.args_schema.model_json_schema())

    llm_output_string = "{\"x\": 5, \"y\": 2}"  # this is the output from the LLM
    llm_output_dict = json.loads(llm_output_string)  # load as dictionary

    res = exponentiate.func(**llm_output_dict)
    print(res)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "you're a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # must align with MessagesPlaceholder variable_name
        return_messages=True  # to return Message objects
    )

    from langchain.agents import create_tool_calling_agent

    tools = [add, subtract, multiply, exponentiate]

    agent = create_tool_calling_agent(
        llm=llm, tools=tools, prompt=prompt
    )

    from langchain.agents import AgentExecutor

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

    agent_executor.invoke({
        "input": "what is 10.7 multiplied by 7.68?",
        "chat_history": memory.chat_memory.messages,
    })


if __name__ == "__main__":
    main()
