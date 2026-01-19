# Insecure Agentic API Examples

This repository contains small, intentionally insecure examples that demonstrate common failure modes in agentic or tool-using LLM APIs. Each script focuses on a specific vulnerability pattern to make it easier to study, reproduce, and harden against in production systems.

## Contents

- `parameter_tampering.py`: A full chat-completions wrapper that forwards arbitrary request parameters to the OpenAI client.
- `raw_sti.py`: A raw tool-call injection example that executes SQL using untrusted tool-call arguments.
- `role_injection.py`: A multi-agent runner that exposes code generation and execution flow to prompt/role injection.
- `unbounded_consumption.py`: A simplified chat API with tool execution that can trigger unbounded tool usage or resource consumption.

## Requirements

- Python 3.10+
- An OpenAI API key (for scripts that call the OpenAI client)
- Optional: PyTorch and Transformers (for `raw_sti.py`)

Suggested dependencies (adapt as needed for your environment):

```
pip install fastapi uvicorn openai httpx pydantic
pip install torch transformers  # for raw_sti.py
```

Set your API key before running the FastAPI examples:

```
export OPENAI_API_KEY="your-key"
```

## Running the FastAPI Examples

Each FastAPI script defines its own app. Use `uvicorn` to run them locally:

```
uvicorn parameter_tampering:app --reload
uvicorn role_injection:app --reload
uvicorn unbounded_consumption:app --reload
```

Then send requests to the exposed endpoints (e.g., `/chat` or `/ask`) with your preferred HTTP client.

# Blogs

For more background on the vulnerability patterns showcased here, check out the following write-ups.

Learn more at:
- [Special Token Injection (STI) Attack Guide](https://blog.sentry.security/special-token-injection-sti-attack-guide/)
