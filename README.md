# Agents Course

Click here to get the course: [Agents Course](https://wandb.me/agents)

A comprehensive course on building AI agents using Python, OpenAI, and Weave. This course covers various aspects of agent development, from basic workflows to complex multi-agent systems with memory and evaluation capabilities.

## Course Structure

The course is organized into several modules:

1. **Basic Workflow** (`0_workflow.py`): Introduction to basic OpenAI and Weave integration
2. **Simple Agent** (`1_agent.py`): Implementation of a minimal agent architecture
3. **Memory & Retrieval** (`2_memory_retrieval.py`): Adding memory capabilities to agents
4. **Multi-Agent Systems** (`3_multi_agents.py`): Building complex systems with multiple specialized agents
5. **Evaluation** (`4_evals.py`): Testing and evaluating agent performance

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended for dependency management)
- OpenAI API key
- Wandb API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agents-course.git
cd agents-course
```

2. Create and activate a virtual environment with all the dependencies:
```bash
uv venv .venv
uv sync
```

3. Set up your environment variables:
Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=your_openai_api_key
WANDB_API_KEY=your_wandb_api_key
```
