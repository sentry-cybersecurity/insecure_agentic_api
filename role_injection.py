from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Any
import logging
import re
import json
import ast
import tempfile

# ---- AGENT DEFINITIONS ----
from agents import Agent, Runner
# Code generation agent (Python or Node.js, NO OS libraries)
code_gen_agent = Agent(
    name="code_gen_agent",
    instructions=(
        "SCOPE: Generate only Python code or Node.js (JavaScript) code that do NOT use any OS libraries "
        "(e.g., 'os', 'sys', 'subprocess' for Python; 'fs', 'child_process', or similar for Node.js). "
        "If the user requests code involving out-of-scope libraries, respond only with: "
        "{'code_allowed': False, 'reason': 'OS libraries are not allowed.'} "
        "Otherwise, respond with: {'code_allowed': True, 'code': '<the code>', 'language': 'python' or 'nodejs'} "
        "No extra words."
    )
)

# Triage agent now supports nodejs
triage_agent = Agent(
    name="triage_agent",
    instructions=(
        "You receive {'code_allowed': True, 'code': '<code>', 'language': 'python' or 'nodejs'} from code_gen_agent. "
        "Handoff to python_agent if language is python, or nodejs_agent if language is nodejs. "
        "Respond ONLY with: {'next_agent': 'python_agent'} or {'next_agent': 'nodejs_agent'}."
    ),
    handoffs=["python_agent", "nodejs_agent"]
)

python_agent = Agent(
    name="python_agent",
    instructions="You run Python code safely and return output.",
)
nodejs_agent = Agent(
    name="nodejs_agent",
    instructions="You run Node.js (JavaScript) code safely and return output.",
)

AGENTS = {
    "code_gen_agent": code_gen_agent,
    "triage_agent": triage_agent,
    "python_agent": python_agent,
    "nodejs_agent": nodejs_agent,
}

# ---- FASTAPI DEFINITIONS ----

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class Message(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    messages: List[Message]

def parse_code_gen(output: str):
    try:
        result = ast.literal_eval(output)
        if not result.get('code_allowed', False):
            return {
                "allowed": False,
                "reason": result.get("reason", "Rejected.")
            }
        return {
            "allowed": True,
            "code": result.get("code"),
            "language": result.get("language")
        }
    except Exception as e:
        return {"allowed": False, "reason": f"Parse error: {e}"}

def parse_next_agent(output: str):
    match = re.search(r"'next_agent':\s*'(\w+)'", output)
    if match:
        return match.group(1)
    return None

# Python code runner
def run_python_code(code):
    try:
        output = []
        def unsafe_print(*args, **kwargs):
            output.append(" ".join(str(arg) for arg in args))
        builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        builtins["print"] = unsafe_print
        print("[DEBUG] About to exec code:")
        print(code)
        exec(code, {"__builtins__": builtins}, {})
        print("[DEBUG] Code execution finished.")
        return {"output": "\n".join(output) if output else "Code executed."}
    except Exception as e:
        return {"output": f"Error: {str(e)}"}

# Node.js code runner
def run_nodejs_code(code):
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as temp_js:
            temp_js.write(code)
            temp_js.flush()
            result = subprocess.run(
                ["node", temp_js.name],
                capture_output=True,
                text=True,
                timeout=5
            )
        return {"output": result.stdout or result.stderr}
    except Exception as e:
        return {"output": f"Error: {str(e)}"}

@app.post("/ask")
async def ask(request: ChatRequest):
    messages = [msg.dict() for msg in request.messages]
    outputs = []

    # Step 1: code_gen_agent - Generate code or reject if out of scope
    result1 = await Runner.run(AGENTS["code_gen_agent"], messages)
    output1 = result1.final_output
    code_gen = parse_code_gen(output1)
    outputs.append({"agent": "code_gen_agent", "output": output1, **code_gen})

    if not code_gen["allowed"]:
        return JSONResponse(content={
            "steps": outputs,
            "final_output": output1,
            "current_agent": "code_gen_agent"
        })

    # Step 2: triage_agent - Decide which executor to use
    handoff_msg = [{
        "role": "system",
        "content": "Handoff for code execution.",
    }, {
        "role": "user",
        "content": output1
    }]
    result2 = await Runner.run(AGENTS["triage_agent"], handoff_msg)
    output2 = result2.final_output
    next_agent = parse_next_agent(output2)
    outputs.append({"agent": "triage_agent", "output": output2, "next_agent": next_agent})

    # Step 3: Run the code if possible
    exec_output = None
    if next_agent == "python_agent":
        exec_output = run_python_code(code_gen["code"])
    elif next_agent == "nodejs_agent":
        exec_output = run_nodejs_code(code_gen["code"])
    else:
        return JSONResponse(content={
            "steps": outputs,
            "final_output": output2,
            "current_agent": "triage_agent"
        })

    outputs.append({"agent": next_agent, "output": exec_output["output"]})

    return JSONResponse(content={
        "steps": outputs,
        "final_output": exec_output["output"],
        "current_agent": next_agent
    })
