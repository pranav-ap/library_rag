from .RetrieverTool import RetrieverTool
from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, LiteLLMModel


class RAGAgent:
    def __init__(self):
        self.agent = CodeAgent(
            tools=[RetrieverTool()],
            model=self._init_model(),
            max_steps=2,
            add_base_tools=False,  # True,
            # planning_interval=3,
            # max_print_outputs_length=1000,
            verbosity_level=0,
        )

    @staticmethod
    def _init_model() -> LiteLLMModel:
        model = LiteLLMModel(
            model_id="ollama_chat/qwen2.5:7b",
            api_base="http://localhost:11434"
        )

        # model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

        return model

    def query(self, user_prompt):
        response = self.agent.run(user_prompt)
        return response
