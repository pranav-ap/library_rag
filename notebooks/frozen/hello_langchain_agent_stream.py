import json
import asyncio
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.runnables import ConfigurableField
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.tools import tool

model_name = "qwen2.5:7b"
llm = ChatOllama(
    temperature=0.0,
    model=model_name
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)


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


@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user.
    The answer should be in natural language as this will be provided
    to the user directly. The tools_used must include a list of tool
    names that were used within the `scratchpad`. You MUST use this tool
    to conclude the interaction.
    """
    return {"answer": answer, "tools_used": tools_used}


tools = [add, multiply, exponentiate, subtract, final_answer]
name2tool = {tool.name: tool.func for tool in tools}

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided back to you. You MUST "
        "then use the final_answer tool to provide a final answer to the user. "
        "DO NOT use the same tool more than once."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


class QueueCallbackHandler(AsyncCallbackHandler):
    """Callback handler that puts tokens into a queue."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue

            token_or_done = await self.queue.get()

            if token_or_done == "<<DONE>>":
                # this means we're done
                return
            if token_or_done:
                yield token_or_done

    async def on_llm_new_token(self, *args, **kwargs) -> None:
        """Put new token in the queue."""
        chunk = kwargs.get("chunk")

        if chunk:
            # check for final_answer tool call
            if tool_calls := chunk.message.additional_kwargs.get("tool_calls"):
                if tool_calls[0]["function"]["name"] == "final_answer":
                    # this will allow the stream to end on the next `on_llm_end` call
                    self.final_answer_seen = True

        self.queue.put_nowait(kwargs.get("chunk"))

        return

    async def on_llm_end(self, *args, **kwargs) -> None:
        """Put None in the queue to signal completion."""
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")

        return


class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, max_iterations: int = 3):
        self.chat_history = []
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

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False) -> dict:
        count = 0
        agent_scratchpad = []
        tool_out = {}

        while count < self.max_iterations:
            async def stream(query: str):
                response = self.agent.with_config(
                    callbacks=[streamer]
                )

                output = None

                async for token in response.astream({
                    "input": query,
                    "chat_history": self.chat_history,
                    "agent_scratchpad": agent_scratchpad
                }):
                    if output is None:
                        output = token
                    else:
                        output += token

                    if token.content != "" and verbose:
                        print(f"content: {token.content}", flush=True)

                    tool_calls = token.additional_kwargs.get("tool_calls")

                    if tool_calls:
                        if verbose:
                            print(f"tool_calls: {tool_calls}", flush=True)

                        tool_name = tool_calls[0]["function"]["name"]
                        if tool_name and verbose:
                            print(f"tool_name: {tool_name}", flush=True)

                        arg = tool_calls[0]["function"]["arguments"]
                        if arg != "" and verbose:
                            print(f"arg: {arg}", flush=True)

                return AIMessage(
                    content=output.content,
                    tool_calls=output.tool_calls,
                    tool_call_id=output.tool_calls[0]["id"]
                )

            tool_call = await stream(query=input)
            if verbose:
                print(f'{tool_call=}', flush=True)

            agent_scratchpad.append(tool_call)

            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_call_id

            tool_out = name2tool[tool_name](**tool_args)

            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )

            agent_scratchpad.append(tool_exec)

            count += 1
            if tool_name == "final_answer":
                break

        final_answer = tool_out["answer"]

        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])

        return tool_out


async def main():
    agent_executor = CustomAgentExecutor()

    queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)

    task = asyncio.create_task(
        agent_executor.invoke(
            input="What is 10 + 10?",
            streamer=streamer,
            verbose=True
        )
    )

    async for token in streamer:
        if token == "<<STEP_END>>":
            print("\n", flush=True)

        elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
            if tool_name := tool_calls[0]["function"]["name"]:
                print(f"Calling {tool_name}...", flush=True)

            if tool_args := tool_calls[0]["function"]["arguments"]:
                print(f"{tool_args}", end="", flush=True)

    _ = await task


asyncio.run(main())
