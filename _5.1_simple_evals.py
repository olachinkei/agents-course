from typing import Any, Callable

import weave
from agents import Agent, Runner, function_tool
from weave import EvaluationLogger

import config

weave.init(project_name=config.WEAVE_PROJECT)


@function_tool
def add(a: int, b: int) -> int:
    return a + b


calc_agent = Agent(
    name="CalcAgent",
    model="gpt-4.1",
    instructions="If asked for arithmetic call `add`, else greet politely.",
    tools=[add],
)


@weave.op()
async def llm_judge(criteria: str) -> Callable[[str, str], bool]:
    """Return an LLM‑based evaluator closure."""

    async def judge(actual: str, expected: str) -> bool:
        judge_agent = Agent(
            name="Judge",
            model="gpt-4o-mini",
            instructions=f"Return True if the output meets: {criteria}.",
            output_type=bool,
        )
        result = await Runner.run(
            judge_agent,
            f"Criteria: {criteria}\nActual: {actual}\nExpected: {expected}",
        )
        return result.final_output

    return judge


@weave.op()
async def run_eval(
    prompt: str,
    expected: Any,
    evaluator: Callable[[Any, Any], bool] = lambda a, e: a == e,
) -> tuple[bool, Any]:
    result = await Runner.run(calc_agent, prompt)
    # If evaluator is async, await it
    if asyncio.iscoroutinefunction(evaluator):
        ok = await evaluator(result.final_output, expected)
    else:
        ok = evaluator(result.final_output, expected)
    return ok, result.final_output


@weave.op()
async def setup_tests():
    base_tests = [
        ("What is 2+3? Respond with ONLY the number", "5"),
        ("hello", "Hello! How can I help you today?"),
    ]
    
    semantic_test = await llm_judge("semantic similarity")
    base_tests.append(("What is the capital of France?", "Paris", semantic_test))
    
    greeting_test = await llm_judge("Does the reply convey a polite greeting?")
    similarity_test = await llm_judge("Is the answer very similar to the expected answer?")
    number_test = await llm_judge("Do the outputs represent the same number?")
    
    base_tests.extend([
        ("hi there!", "Hello! How can I help you today?", greeting_test),
        ("What's the capital of Germany?", "Berlin", similarity_test),
        ("7 + 8, but spell it out", "fifteen", number_test),
    ])
    
    return base_tests


@weave.op()
async def chapter_5_point_1_simple_evals():
    TESTS = await setup_tests()
    eval_logger = EvaluationLogger(model="CalcAgent", dataset="Simple Calculator Tests")

    passed = 0
    for i, (prompt, expected, *ev) in enumerate(TESTS, 1):
        ok, got = await run_eval(prompt, expected, ev[0] if ev else (lambda a, e: a == e))
        passed += ok

        pred_logger = eval_logger.log_prediction(inputs={"prompt": prompt}, output=got)

        pred_logger.log_score(scorer="correctness", score=float(ok))

        pred_logger.finish()

        print(f"{'✅' if ok else '❌'} {i}. {prompt!r} → {got!r}")

    eval_logger.log_summary(
        {
            "total_tests": len(TESTS),
            "passed_tests": passed,
            "pass_rate": passed / len(TESTS),
        }
    )

    print(f"\nPassed {passed}/{len(TESTS)} tests")


if __name__ == "__main__":
    import asyncio
    asyncio.run(chapter_5_point_1_simple_evals())
