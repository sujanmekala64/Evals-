import json
import re

from llm import invoke_llm

with open(
    "data/knowledge_base.txt",
    "r"
) as f:
    text = f.read()

prompt = f"""
Generate 20 question answer pairs.

Return ONLY JSON.

TEXT:
{text}
"""

response = invoke_llm(prompt)

match = re.search(
    r"\[.*\]",
    response,
    re.DOTALL
)

if not match:
    raise Exception(
        "JSON not found"
    )

dataset = json.loads(
    match.group()
)

with open(
    "data/generated_testset.json",
    "w"
) as f:
    json.dump(
        dataset,
        f,
        indent=2
    )

print(
    "Generated",
    len(dataset),
    "questions"
)