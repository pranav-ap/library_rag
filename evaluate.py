from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    PromptAlignmentMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

from src import RAGSystem


def generate_test_cases():
    rag_system = RAGSystem()

    user_prompts = [
        "What is Sherlock's assistant's name?",
        "What is Sherlock's home address'?",
        "What is Sherlock's enemy's name?",
        "What role does Dr. John Watson play in Sherlock Holmes' investigations?",
        "How does Sherlock Holmes' addiction to drugs, like cocaine, affect his personality and his approach to solving cases?",
        "What is the significance of the violin in Sherlock Holmes' life?",
        "Why is Moriarty considered his archenemy?",
    ]

    test_cases = []

    for user_prompt in user_prompts:
        response, retrieval_results = rag_system.query_test(user_prompt)

        test_case = LLMTestCase(
            input=user_prompt,
            actual_output=response,
            retrieval_context=retrieval_results["snippets"],
        )

        test_cases.append(test_case)

    return test_cases


def run_eval():
    test_cases = generate_test_cases()

    preferences = [
        'Answer in simple words within 100 words. Omit supplementary details.',
        'Only use context information. Nothing else.',
    ]

    answer_relevancy_metric = AnswerRelevancyMetric()
    faithfulness_metric = FaithfulnessMetric()
    contextual_relevancy_metric = ContextualRelevancyMetric()
    prompt_alignment_metric = PromptAlignmentMetric(
        prompt_instructions=preferences,
    )

    evaluate(
        test_cases=test_cases,
        metrics=[
            answer_relevancy_metric,
            faithfulness_metric,
            contextual_relevancy_metric,
            prompt_alignment_metric
        ],
        skip_on_missing_params=True,
    )


if __name__ == "__main__":
    run_eval()
