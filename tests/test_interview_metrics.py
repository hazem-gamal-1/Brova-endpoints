import pytest
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ToxicityMetric,
    BiasMetric,
    GEval,
)

# Set a dummy API key for evaluation if not present in the environment
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-placeholder"

def test_interview_response_quality():
    """
    Test suite for evaluating the Brova Interview Agent using 15 DeepEval metrics.
    These metrics assess the quality, safety, and relevance of the agent's responses.
    """
    
    # ---------------------------------------------------------
    # 1. Define the Test Case Data
    # ---------------------------------------------------------
    # In a real test, these values would be populated dynamically by running the 
    # Brova agent (api.interview.Interview) and extracting the results.
    input_text = "Start the interview for this candidate based on the provided CV."
    actual_output = "Hello! I am Brova, an AI interviewer. Could you tell me about your experience with Python at your last job?"
    
    # Context retrieved from vector DB or document loaders (e.g., the parsed CV)
    retrieval_context = [
        "Candidate has 5 years of Python experience.", 
        "Candidate worked as a Senior Backend Engineer at TechCorp."
    ]
    
    # Expected ideal output for the prompt
    expected_output = "Hello! I am Brova, an AI interviewer. Could you tell me about your experience with Python?"
    
    # The factual context for hallucination/faithfulness checks
    context = [
        "Candidate has 5 years of Python experience.", 
        "Candidate applied for AI Engineer role."
    ]

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
        context=context,
    )

    # ---------------------------------------------------------
    # 2. Initialize Standard Metrics (1-6 & 10-11)
    # ---------------------------------------------------------
    
    # 1. Faithfulness -> grounded in CV/job context, no hallucination
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)

    # 2. Answer Relevancy -> directly answers the question
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

    # 3. Hallucination -> detects fabricated information
    hallucination_metric = HallucinationMetric(threshold=0.5)

    # 4. Contextual Precision -> relevance of retrieved context
    contextual_precision_metric = ContextualPrecisionMetric(threshold=0.5)

    # 5. Contextual Recall -> missing important context check
    contextual_recall_metric = ContextualRecallMetric(threshold=0.5)

    # 6. Contextual Relevancy -> usefulness of context overall
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.5)

    # 10. Toxicity -> detects unsafe or inappropriate output
    toxicity_metric = ToxicityMetric(threshold=0.5)
    
    # 11. Bias -> detects gender, racial, or other bias
    bias_metric = BiasMetric(threshold=0.5)

    # ---------------------------------------------------------
    # 3. Initialize Custom GEval Metrics (7-9 & 12-15)
    # ---------------------------------------------------------

    # 7. GEval Interview Quality -> interview flow + question quality
    interview_quality_metric = GEval(
        name="Interview Quality",
        criteria="Evaluate if the interview flow is natural, and if the question quality is high and relevant to the candidate's background.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
        threshold=0.5
    )

    # 8. GEval Code Quality -> correctness and improvement of code
    code_quality_metric = GEval(
        name="Code Quality",
        criteria="Evaluate if the provided code improvements (if any) are logically correct, optimal, and properly justified without writing raw code blocks in the conversation.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    # 9. GEval Instruction Following -> follows system rules and format
    instruction_following_metric = GEval(
        name="Instruction Following",
        criteria="Evaluate if the AI follows its strict system rules, such as not exposing its internal interview plan/todos to the user.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    # 12. GEval Persona Consistency -> stays in character
    persona_consistency_metric = GEval(
        name="Persona Consistency",
        criteria="Evaluate if the AI maintains its assigned personality consistently throughout the response.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    # 13. GEval Language Consistency -> adheres to requested language
    language_consistency_metric = GEval(
        name="Language Consistency",
        criteria="Evaluate if the AI responds in the exact requested language (e.g., Egyptian Arabic with English tech terms) without arbitrarily switching.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    # 14. GEval Plan Adherence -> follows the generated plan
    plan_adherence_metric = GEval(
        name="Plan Adherence",
        criteria="Evaluate if the AI's current question and behavior align logically with a structured interview plan.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    # 15. GEval Feedback Quality -> evaluates the final summary feedback
    feedback_quality_metric = GEval(
        name="Feedback Quality",
        criteria="Evaluate if the final feedback provided to the candidate is constructive, measurable, and free of subjective bias.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    # ---------------------------------------------------------
    # 4. Execute the Tests
    # ---------------------------------------------------------
    
    metrics = [
        faithfulness_metric,
        answer_relevancy_metric,
        hallucination_metric,
        contextual_precision_metric,
        contextual_recall_metric,
        contextual_relevancy_metric,
        interview_quality_metric,
        code_quality_metric,
        instruction_following_metric,
        toxicity_metric,
        bias_metric,
        persona_consistency_metric,
        language_consistency_metric,
        plan_adherence_metric,
        feedback_quality_metric
    ]

    # Run the assertions for all metrics
    assert_test(test_case, metrics)

if __name__ == "__main__":
    # Automatic report generation when running the file directly
    pytest.main(["-v", f"--html={os.path.basename(__file__).replace('.py', '_report.html')}", "--self-contained-html", __file__])
