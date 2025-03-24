def main():
    from src import setup_env_variables
    setup_env_variables()

    from src import setup_workflow
    workflow = setup_workflow()
    graph = workflow.compile()

    inputs = {
        "question": "When did Napoleon born?",
        "max_retries": 3,
        "MAX_ANSWER_LENGTH": 150,
    }

    results = []
    for event in graph.stream(inputs, stream_mode="values"):
        results.append(event)

    if len(results):
        print(results[-1]['answer'])


if __name__ == "__main__":
    main()
