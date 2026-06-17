"""Server-Sent Events streaming for real-time agent output."""

import json
import asyncio
from typing import AsyncGenerator


async def stream_agent_response(
    agent, state: dict, config: dict
) -> AsyncGenerator[str, None]:
    """
    Stream agent execution as SSE events.
    
    Event types:
      - step: agent node started/completed
      - token: streamed output token
      - done: execution complete
      - error: execution failed
    """
    try:
        async for event in agent.astream_events(state, config, version="v2"):
            kind = event.get("event", "")

            if kind == "on_chain_start":
                name = event.get("name", "")
                yield f"data: {json.dumps({'type': 'step', 'agent': name, 'status': 'started'})}\n\n"

            elif kind == "on_chain_end":
                name = event.get("name", "")
                yield f"data: {json.dumps({'type': 'step', 'agent': name, 'status': 'done'})}\n\n"

            elif kind == "on_llm_stream":
                chunk = event.get("data", {}).get("chunk", "")
                if chunk:
                    content = getattr(chunk, "content", str(chunk))
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        await asyncio.sleep(0.01)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"