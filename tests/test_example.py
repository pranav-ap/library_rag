import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    PromptAlignmentMetric,
    ContextualRelevancyMetric,
)
from src import RAGAgent


def generate_eval_dataset():
    agent = RAGAgent()

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
        response = agent.query(user_prompt)

        test_case = LLMTestCase(
            input=user_prompt,
            actual_output=response,
            # retrieval_context=retrieval_results["snippets"],
        )

        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)
    return dataset


@pytest.mark.parametrize(
    "test_case",
    generate_eval_dataset(),
)
def test_correctness(test_case: LLMTestCase):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )

    assert_test(test_case, [correctness_metric])


@pytest.mark.parametrize(
    "test_case",
    generate_eval_dataset(),
)
def test_everything(test_case: LLMTestCase):
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

    assert_test(test_case, [relevancy_metric])


