from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler()


def main():
    from src import setup_env_variables
    setup_env_variables()

    from src import setup_workflow
    workflow = setup_workflow()
    graph = workflow.compile()

    print("RAG is ready. Type 'q' to quit.")

    while True:
        question = input("\nAsk a question: ")
        if question.lower() in ["q", "exit"]:
            print("Goodbye!")
            break

        inputs = {
            "question": question,
            "max_retries": 3,
            "MAX_ANSWER_LENGTH": 150,
        }

        results = []
        for event in graph.stream(inputs, config={"callbacks": [langfuse_handler]}):
            results.append(event)

        if results:
            print("\nAnswer:", results[-1]['answer'])


if __name__ == "__main__":
    main()
