import asyncio

import weave
from agents import Agent, Runner, function_tool, set_trace_processors

import config
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

weave.init(project_name=config.WEAVE_PROJECT)
set_trace_processors([WeaveTracingProcessor()])

@function_tool
def add(a: int, b: int) -> int:
    return a + b


agent = Agent(
    name="Calculator",
    instructions="You are a calculator. You can add two numbers together.",
    tools=[add],
)

@weave.op()
async def run_sdk(input):
    response = await Runner.run(agent, input)
    print(response.final_output)
    return response.final_output


if __name__ == "__main__":
    asyncio.run(run_sdk(input="What is 2 + 2?"))
