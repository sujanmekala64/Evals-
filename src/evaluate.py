from metrics import (
    semantic_similarity,
    evaluate_completeness,
    evaluate_relevance,
    evaluate_faithfulness,
    generate_paraphrases
)

from rag_pipeline import answer_question

import json
import pandas as pd
import numpy as np

with open(
    "data/generated_testset.json"
) as f:

    dataset = json.load(f)

results = []

for sample in dataset:

    question = sample["question"]

    ground_truth = sample["answer"]

    answer, context = answer_question(
        question
    )

    #######################################
    # CORRECTNESS
    #######################################

    correctness = semantic_similarity(
        answer,
        ground_truth
    )

    #######################################
    # COMPLETENESS
    #######################################

    completeness = evaluate_completeness(
        question,
        answer,
        ground_truth
    )

    #######################################
    # RELEVANCE
    #######################################

    relevance = evaluate_relevance(
        question,
        answer
    )

    #######################################
    # FAITHFULNESS
    #######################################

    faithfulness = evaluate_faithfulness(
        answer,
        context
    )

    #######################################
    # ROBUSTNESS
    #######################################

    paraphrases = generate_paraphrases(
        question
    )

    robustness_scores = []

    for p in paraphrases:

        alt_answer, _ = answer_question(
            p
        )

        robustness_scores.append(
            semantic_similarity(
                answer,
                alt_answer
            )
        )

    robustness = np.mean(
        robustness_scores
    )

    #######################################
    # RETRIEVAL ACCURACY
    #######################################

    retrieval_accuracy = (
        semantic_similarity(
            context,
            ground_truth
        )
    )

    results.append({

        "question": question,

        "correctness":
        correctness,

        "completeness":
        completeness,

        "relevance":
        relevance,

        "faithfulness":
        faithfulness,

        "robustness":
        robustness,

        "retrieval_accuracy":
        retrieval_accuracy
    })

df = pd.DataFrame(results)

print("\n========== FINAL REPORT ==========\n")

for column in [
    "correctness",
    "completeness",
    "relevance",
    "faithfulness",
    "robustness",
    "retrieval_accuracy"
]:

    print(
        f"{column}: "
        f"{df[column].mean():.3f}"
    )

hallucination_rate = (
    10 - df["faithfulness"].mean()
) / 10

print(
    f"\nHallucination Rate: "
    f"{hallucination_rate:.3f}"
)

df.to_csv(
    "results/rag_evaluation_results.csv",
    index=False
)