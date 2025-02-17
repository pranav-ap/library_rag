from utils import logger

from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

import gradio as gr
from src import RAGAgent


def setup_tracing():
    # Go to http://localhost:6006/
    tracer_provider = register(
      project_name="library-rag-app",
      endpoint="http://localhost:4317",
    )

    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)


def text_main():
    user_prompt = "Where does Sherlock Holmes live? Check local sources"
    logger.info('User Prompt')
    print(user_prompt)

    agent = RAGAgent()
    response = agent.query(user_prompt)

    logger.info('Response')
    print(response)


def ui_main():
    agent = RAGAgent()

    def query_rag_agent(user_prompt):
        response = agent.query(user_prompt)
        return response

    iface = gr.Interface(
        fn=query_rag_agent,
        inputs=gr.Textbox(lines=2, placeholder="Enter your query..."),
        outputs=gr.Textbox(),
        title="RAG",
        description="This tool retrieves relevant snippets from a local knowledge base to answer your queries.",
    )

    iface.launch()


if __name__ == "__main__":
    setup_tracing()
    ui_main()
