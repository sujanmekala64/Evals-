from langchain_community.vectorstores import FAISS

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from embeddings import embeddings

with open(
    "data/knowledge_base.txt",
    "r"
) as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

vectorstore = FAISS.from_texts(
    chunks,
    embeddings
)

vectorstore.save_local(
    "vectorstore/faiss_index"
)

print(
    f"Stored {len(chunks)} chunks"
)