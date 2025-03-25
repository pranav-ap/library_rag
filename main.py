def main():
    from src import make_rag
    rag, langfuse_handler = make_rag()

    print("RAG is ready. Type 'q' to quit.")

    while True:
        question = input("\nUser : ")
        if question.lower() in ["q", "exit"]:
            print("Goodbye!")
            break

        inputs = {
            "question": question,
            "max_retries": 3,
            "MAX_ANSWER_LENGTH": 150,
        }

        results = []
        for event in rag.graph.stream(inputs, config={"callbacks": [langfuse_handler]}):
            results.append(event)

        if results:
            print("\nAI :", results[-1]['generate_answer']['answer'])


if __name__ == "__main__":
    main()
