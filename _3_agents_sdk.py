import asyncio

import weave
from agents import Agent, Runner, function_tool

import config

weave.init(project_name=config.WEAVE_PROJECT)


@function_tool
def add(a: int, b: int) -> int:
    return a + b


agent = Agent(
    name="Calculator",
    instructions="You are a calculator. You can add two numbers together.",
    tools=[add],
)


@weave.op()
async def chapter_3_agents_sdk():
    response = await Runner.run(agent, "What is 2 + 2?")
    print(response.final_output)


if __name__ == "__main__":
    asyncio.run(chapter_3_agents_sdk())
