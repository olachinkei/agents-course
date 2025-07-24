from typing import Any, Callable

import weave
from agents import Agent, Runner, function_tool, set_trace_processors
from weave import EvaluationLogger

import config
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor
set_trace_processors([WeaveTracingProcessor()])

weave.init(project_name=config.WEAVE_PROJECT)


@function_tool
def add(a: int, b: int) -> int:
    return a + b


@function_tool
def subtract(a: int, b: int) -> int:
    return a - b


@function_tool
def multiply(a: int, b: int) -> int:
    return a * b


@function_tool
def divide(a: int, b: int) -> float:
    return a / b


calc_agent = Agent(
    name="CalcAgent",
    model="gpt-4.1",
    instructions="You are a calculator. Use the provided tools to perform arithmetic operations (add, subtract, multiply, divide). If you can perform the calculation, return only the numeric result. If the calculation is not possible with your available tools, respond with 'Cannot calculate'.",
    tools=[add, subtract, multiply, divide],
)


@weave.op()
def calculate_score(model_output: Any, expected: Any) -> bool:
    """Calculate correctness score for the prediction"""
    # Handle type conversion for numeric comparisons
    try:
        # Try to convert both to float for numeric comparison
        if isinstance(expected, (int, float)) and isinstance(model_output, str):
            # If expected is numeric and model_output is string, try to convert
            if model_output.replace('.', '').replace('-', '').isdigit():
                model_output = float(model_output) if '.' in model_output else int(model_output)
        elif isinstance(expected, (int, float)) and isinstance(model_output, (int, float)):
            # Both are numeric, handle int/float comparison
            if isinstance(expected, int) and isinstance(model_output, float) and model_output.is_integer():
                model_output = int(model_output)
            elif isinstance(expected, float) and isinstance(model_output, int):
                expected = float(expected)
    except (ValueError, AttributeError):
        pass  # If conversion fails, use original values
    
    print(f"Comparing: {model_output} (type: {type(model_output)}) == {expected} (type: {type(expected)})")
    return model_output == expected


@weave.op()
async def user_model(prompt: str) -> Any:
    """Model that performs arithmetic operations based on natural language prompts"""
    result = await Runner.run(calc_agent, prompt)
    return result.final_output


@weave.op()
async def simple_evals():
    # Initialize EvaluationLogger BEFORE calling the model
    eval_logger = EvaluationLogger(
        model="CalcAgent",
        dataset="Simple Math Operations"
    )

    # Example input data with various natural language calculation requests
    eval_samples = [
        # Basic calculations
        {'inputs': {'prompt': 'What is 2 plus 3?'}, 'expected': 5},
        {'inputs': {'prompt': 'Calculate 10 minus 4'}, 'expected': 6},
        {'inputs': {'prompt': 'Can you multiply 3 by 7?'}, 'expected': 21},
        {'inputs': {'prompt': 'What is 15 divided by 3?'}, 'expected': 5.0},
        {'inputs': {'prompt': 'Add 100 and 200 together'}, 'expected': 300},
        {'inputs': {'prompt': 'What is 8 times 9?'}, 'expected': 72},
        {'inputs': {'prompt': 'Subtract 25 from 50'}, 'expected': 25},
        
        # Impossible calculations (should return "Cannot calculate")
        {'inputs': {'prompt': 'What is the square root of 16?'}, 'expected': 'Cannot calculate'},
        {'inputs': {'prompt': 'Calculate 2 to the power of 3'}, 'expected': 'Cannot calculate'},
        {'inputs': {'prompt': 'What is the sine of 90 degrees?'}, 'expected': 'Cannot calculate'},
        
        # More challenging problems
        {'inputs': {'prompt': 'What is 1247 multiplied by 863?'}, 'expected': 1076161},
        {'inputs': {'prompt': 'Calculate 9876 divided by 1234'}, 'expected': 8.0},
        {'inputs': {'prompt': 'What is 50000 minus 37892?'}, 'expected': 12108},
        {'inputs': {'prompt': 'Add 99999 and 88888'}, 'expected': 188887},
    ]

    passed = 0
    total = len(eval_samples)

    # Iterate through examples, predict, and log
    for i, sample in enumerate(eval_samples, 1):
        inputs = sample["inputs"]
        model_output = await user_model(**inputs)  # Pass inputs as kwargs
        expected = sample["expected"]
        inputs["expected"] = expected  # Add expected value to inputs for logging

        # Log the prediction input and output
        pred_logger = eval_logger.log_prediction(
            inputs=inputs,
            output=model_output
        )

        # Calculate and log a score for this prediction
        correctness_score = calculate_score(model_output, expected)
        if correctness_score:
            passed += 1

        pred_logger.log_score(
            scorer="correctness",  # Simple string name for the scorer
            score=correctness_score
        )

        # Finish logging for this specific prediction
        pred_logger.finish()

        # Print result
        print(f"{'✅' if correctness_score else '❌'} {i}. {inputs['prompt']} → {model_output} (expected: {expected})")

    # Log a final summary for the entire evaluation
    pass_rate = passed / total
    summary_stats = {
        "total_tests": total,
        "passed_tests": passed,
        "pass_rate": pass_rate
    }
    eval_logger.log_summary(summary_stats)

    print(f"\nPassed {passed}/{total} tests (Pass rate: {pass_rate:.2%})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(simple_evals())
