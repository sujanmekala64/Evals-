from langchain_community.vectorstores import FAISS

from embeddings import embeddings
from llm import invoke_llm

vectorstore = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


def answer_question(question):

    docs = retriever.invoke(question)

    context = "\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    prompt = f"""
Answer ONLY from context.

Context:
{context}

Question:
{question}
"""

    answer = invoke_llm(prompt)

    return answer, context