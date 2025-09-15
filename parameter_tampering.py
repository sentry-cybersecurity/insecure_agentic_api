import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from openai import OpenAI

app = FastAPI(title="Full Chat Completions API Wrapper")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

###############
# Request Models
###############

class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None  # deprecated
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, Dict[str, int]]] = None
    user: Optional[str] = None  # deprecated, replaced by prompt_cache_key
    prompt_cache_key: Optional[str] = None
    audio: Optional[Dict[str, Any]] = None
    modalities: Optional[List[str]] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    stream_options: Optional[Dict[str, Any]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None  # deprecated
    functions: Optional[List[Dict[str, Any]]] = None  # deprecated
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    safety_identifier: Optional[str] = None
    store: Optional[bool] = None
    seed: Optional[int] = None  # deprecated
    service_tier: Optional[str] = None
    verbosity: Optional[str] = None
    top_logprobs: Optional[int] = None
    logprobs: Optional[bool] = None
    web_search_options: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None

###############
# Response Models
###############

class ChatResponse(BaseModel):
    raw_response: Dict[str, Any]

###############
# Endpoint
###############

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        params = request.dict(exclude_none=True)
        resp = client.chat.completions.create(**params)
        return ChatResponse(raw_response=resp.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
