import json
import os

import numpy as np
import weave
from agents import Agent, FileSearchTool, function_tool
from openai import OpenAI

import config

weave.init(project_name=config.WEAVE_PROJECT)
client = OpenAI()
MEMORY_FILE = "memory_store.jsonl"


def get_embedding(text: str) -> np.ndarray:
    return np.array(
        client.embeddings.create(model="text-embedding-3-small", input=[text])
        .data[0]
        .embedding
    )


def similarity_from_embeddings(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def read_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_file(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_memories() -> list[dict]:
    return [json.loads(x) for x in read_file(MEMORY_FILE)]


def append_memory(memory: str) -> None:
    write_file(
        MEMORY_FILE,
        json.dumps(
            {"memory": memory.strip(), "embedding": get_embedding(memory).tolist()}
        ),
    )


def relevant_memories(query: str, threshold: float = 0.5) -> list[str]:
    q_emb = get_embedding(query)
    return [
        m["memory"]
        for m in load_memories()
        if similarity_from_embeddings(q_emb, np.array(m["embedding"])) > threshold
    ]


@function_tool
def save_memory(memory: str) -> str:
    append_memory(memory)
    return f"Memory saved: {memory.strip()}"


@function_tool
def query_memory(query: str) -> str:
    hits = [
        m["memory"] for m in load_memories() if query.lower() in m["memory"].lower()
    ]
    return "\n".join(hits) or "No matching memories."


if __name__ == "__main__":
    agent = Agent(
        instructions="Answer questions about the user's memory.",
        tools=[FileSearchTool(vector_store_ids=["123"])],
        model="gpt-4o-mini",
    )
