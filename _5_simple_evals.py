from typing import Any, Callable
from agents import Agent, Runner, function_tool


@function_tool
def add(a: int, b: int) -> int:
    return a + b


calc_agent = Agent(
    name="CalcAgent",
    model="gpt-4.1",
    instructions="If asked for arithmetic call `add`, else greet politely.",
    tools=[add],
)


def llm_judge(criteria: str) -> Callable[[str, str], bool]:
    """Return an LLM‑based evaluator closure."""

    def judge(actual: str, expected: str) -> bool:
        judge_agent = Agent(
            name="Judge",
            model="gpt-4o-mini",
            instructions=f"Return True if the output meets: {criteria}.",
            output_type=bool,
        )
        return Runner.run_sync(
            judge_agent,
            f"Criteria: {criteria}\nActual: {actual}\nExpected: {expected}",
        ).final_output

    return judge


def run_eval(
    prompt: str,
    expected: Any,
    evaluator: Callable[[Any, Any], bool] = lambda a, e: a == e,
) -> tuple[bool, Any]:
    out = Runner.run_sync(calc_agent, prompt).final_output
    return evaluator(out, expected), out


TESTS = [
    ("What is 2+3? Respond with ONLY the number", "5"),
    ("hello", "Hello! How can I help you today?"),
    ("What is the capital of France?", "Paris", llm_judge("semantic similarity")),
]

TESTS += [
    (
        "hi there!",
        "Hello! How can I help you today?",
        llm_judge("Does the reply convey a polite greeting?"),
    ),
    (
        "What's the capital of Germany?",
        "Berlin",
        llm_judge("Is the answer very similar to the expected answer?"),
    ),
    (
        "7 + 8, but spell it out",
        "fifteen",
        llm_judge("Do the outputs represent the same number?"),
    ),
]

if __name__ == "__main__":
    passed = 0
    for i, (prompt, expected, *ev) in enumerate(TESTS, 1):
        ok, got = run_eval(prompt, expected, ev[0] if ev else (lambda a, e: a == e))
        passed += ok
        print(f"{'✅' if ok else '❌'} {i}. {prompt!r} → {got!r}")
    print(f"\nPassed {passed}/{len(TESTS)} tests")
