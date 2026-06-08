from llm import invoke_llm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def semantic_similarity(a, b):

    e1 = embedding_model.encode([a])
    e2 = embedding_model.encode([b])

    return float(
        cosine_similarity(e1, e2)[0][0]
    )


def llm_score(prompt):

    result = invoke_llm(prompt)

    try:
        return float(result.strip())
    except:
        return 0.0


def evaluate_completeness(
    question,
    answer,
    ground_truth
):

    prompt = f"""
Rate completeness from 0 to 10.

Question:
{question}

Ground Truth:
{ground_truth}

Answer:
{answer}

Return ONLY a number.
"""

    return llm_score(prompt)


def evaluate_relevance(
    question,
    answer
):

    prompt = f"""
Rate relevance from 0 to 10.

Question:
{question}

Answer:
{answer}

Return ONLY a number.
"""

    return llm_score(prompt)


def evaluate_faithfulness(
    answer,
    context
):

    prompt = f"""
Rate faithfulness from 0 to 10.

Context:
{context}

Answer:
{answer}

Return ONLY a number.
"""
    return llm_score(prompt)

def generate_paraphrases(question):

    prompt = f"""
Generate 3 paraphrases.

Question:
{question}

Return one per line.
"""

    response = invoke_llm(prompt)

    return [
        x.strip("- ")
        for x in response.split("\n")
        if x.strip()
    ][:3]