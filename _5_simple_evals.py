from typing import Any, Callable, List, Tuple

from agents import Agent, Runner, function_tool
import config


@function_tool
def add(a: int, b: int) -> int:
    return a + b


calc_agent = Agent(
    name="Agent",
    model="gpt-4.1",
    instructions=(
        "If the user asks for arithmetic, call the `add` tool. "
        "Otherwise greet politely."
    ),
    tools=[add],
)


def run_eval(
    agent: Agent,
    user_input: str,
    expected_output: Any,
    evaluator: Callable[[Any, Any], bool] = lambda x, y: x == y,
):
    result = Runner.run_sync(agent, user_input)
    ok = evaluator(result.final_output, expected_output)
    return ok, result.final_output


TESTS: List[Tuple[str, Any]] = [
    ("What is 2+3? Respond with ONLY the number", "5"),
    ("hello", "Hello! How can I help you today?"),
]

if __name__ == "__main__":
    passed = 0
    for i, (prompt, gold) in enumerate(TESTS, 1):
        ok, got = run_eval(calc_agent, prompt, gold)
        passed += ok
        print(f"{'✅' if ok else '❌'} {i}. {prompt!r} → {got!r}")
    print(f"\nPassed {passed}/{len(TESTS)} tests")
