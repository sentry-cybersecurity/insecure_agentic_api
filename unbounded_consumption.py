import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI

app = FastAPI(title="Simplified Chat API (Messages Only)")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

###############
# Request Models
###############

class Message(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_call_id: Optional[str] = None  # for feeding tool responses


class ChatRequest(BaseModel):
    messages: List[Message]


###############
# Response Models
###############

class ChatResponse(BaseModel):
    raw_response: Dict[str, Any]
    full_conversation: List[Dict[str, Any]]


###############
# Tool Implementations
###############

async def http_get_request(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    """Executes a GET request and returns the body text (truncated if very large)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers or {})
        return response.text[:2000]  # truncate to avoid exceeding token limits


tool_registry = {
    "http_get_request": http_get_request
}


###############
# Endpoint
###############

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Always use fixed model and other defaults
        resp = client.chat.completions.create(
            model="gpt-4.1",  # fixed model
            messages=[m.dict(exclude_none=True) for m in request.messages]
        )

        # Start conversation with existing messages
        conversation = [m.dict(exclude_none=True) for m in request.messages]

        # Add assistant message
        assistant_msg = resp.choices[0].message.model_dump()
        conversation.append(assistant_msg)

        # Handle tool calls if present
        if assistant_msg.get("tool_calls"):
            tool_calls = assistant_msg["tool_calls"]

            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])

                if fn_name in tool_registry:
                    result = await tool_registry[fn_name](**args)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    }
                    conversation.append(tool_msg)

            # Feed tool responses back
            follow_up = client.chat.completions.create(
                model="gpt-4.1",
                messages=conversation
            )
            final_msg = follow_up.choices[0].message.model_dump()
            conversation.append(final_msg)

            return ChatResponse(
                raw_response=follow_up.model_dump(),
                full_conversation=conversation
            )

        # No tool call, just return response
        return ChatResponse(
            raw_response=resp.model_dump(),
            full_conversation=conversation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
