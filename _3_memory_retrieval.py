import asyncio
import json
import os

import numpy as np

# import weave
from agents import Agent, Runner, function_tool, FileSearchTool
from openai import OpenAI

import config

VECTOR_STORE_ID = None
MEMORY_VECTOR_FILE_ID = None

# weave.init(project_name=config.WEAVE_PROJECT)
client = OpenAI()
MEMORY_FILE = "memory_store.jsonl"


# @weave.op()
def get_embedding(text: str) -> np.ndarray:
    return np.array(
        client.embeddings.create(model="text-embedding-3-small", input=[text])
        .data[0]
        .embedding
    )


# @weave.op()
def similarity_from_embeddings(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


# @weave.op()
def read_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# @weave.op()
def write_file(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# @weave.op()
def load_memories() -> list[dict]:
    return [json.loads(x) for x in read_file(MEMORY_FILE)]


# @weave.op()
def append_memory(memory: str) -> None:
    write_file(
        MEMORY_FILE,
        json.dumps(
            {"memory": memory.strip(), "embedding": get_embedding(memory).tolist()}
        ),
    )


# @weave.op()
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


# Create memory management agent with memory-related tools
memory_agent_1 = Agent(
    name="Memory Manager 1",
    instructions="Help users save and query their memories.",
    tools=[save_memory, query_memory],
    model="gpt-4o-mini",
)


def get_vector_store_id():
    global VECTOR_STORE_ID
    if VECTOR_STORE_ID:
        return VECTOR_STORE_ID

    # Otherwise, create a new vector store
    vector_store = client.vector_stores.create(
        name="Agent Course",
    )
    client.vector_stores.files.upload_and_poll(
        vector_store_id=vector_store.id,
        file=open("sample_vector_store_memory.txt", "rb"),
    )
    print(f"Created new vector store: {vector_store.id}")
    VECTOR_STORE_ID = vector_store.id
    return vector_store.id


# Create search agent with just FileSearchTool
memory_agent_2 = Agent(
    name="Memory Manager 2",
    instructions="Help search through files.",
    tools=[FileSearchTool(vector_store_ids=[get_vector_store_id()])],
    model="gpt-4o-mini",
)


async def main():
    # First, let's store some memories
    print("Storing memories...")
    response = await Runner.run(
        memory_agent_1,
        "Please save this memory: I learned about Python async/await in March 2024",
    )
    print(response.final_output)

    # Now let's query those memories
    print("\nQuerying memories...")
    response = await Runner.run(memory_agent_1, "What do you remember about Python?")
    print(response.final_output)

    # Use the search agent to look through files
    # print("\nSearching files...")
    response = await Runner.run(
        memory_agent_2, "How was the weather during spring break?"
    )
    print(response.final_output)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
