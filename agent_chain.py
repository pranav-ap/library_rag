import json
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.tools import tool


@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user.
    The answer should be in natural language as this will be provided
    to the user directly. The tools_used must include a list of tool
    names that were used within the `scratchpad`.
    """
    result = {"answer": answer, "tools_used": tools_used}
    return json.dumps(result)


tools = [final_answer]
name2tool = {tool.name: tool.func for tool in tools}


class RAGAgentExecutor:
    def __init__(self, max_iterations: int = 3):
        local_llm = "llama3.2:3b-instruct-fp16"
        llm = ChatOllama(temperature=0.0, model=local_llm)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You're a helpful assistant. When answering a user's question "
                "you should first use one of the tools provided. After using a "
                "tool the tool output will be provided back to you. You MUST "
                "then use the final_answer tool to provide a final answer to the user. "
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.chat_history: [BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent: RunnableSerializable = (
                {
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x["chat_history"],
                    "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
                }
                | prompt
                | llm.bind_tools(tools, tool_choice="any")
        )

    @staticmethod
    def run_tool(count, tool_call):
        tool_name = tool_call.tool_calls[0]["name"]
        tool_args = tool_call.tool_calls[0]["args"]
        tool_call_id = tool_call.tool_calls[0]["id"]

        print(f"{count}: {tool_name}({tool_args})")
        tool_output = name2tool[tool_name](**tool_args)

        tool_msg = ToolMessage(
            content=f"The {tool_name} tool returned {tool_output}.",
            tool_call_id=tool_call_id,
        )

        return tool_name, tool_output, tool_msg

    def invoke(self, query: str) -> dict | None:
        count = 0
        agent_scratchpad = []
        tool_output = ''

        while count < self.max_iterations:
            tool_call = self.agent.invoke({
                "input": query,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })

            tool_name, tool_output, tool_msg = self.run_tool(count, tool_call)
            agent_scratchpad.extend([tool_call, tool_msg])
            count += 1

            if tool_name == "final_answer":
                break

        self.chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=tool_output)
        ])

        try:
            res = json.loads(tool_output)
            return res

        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print(f"{tool_output=}")
            return None

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


agent_executor = RAGAgentExecutor()
res = agent_executor.invoke(query="What is 10 + 10?")
print(f'{res=}')

