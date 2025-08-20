# api.py
import os
import asyncio
import logging
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from google.adk.memory import InMemoryMemoryService

load_dotenv()
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chat-api")

# Initialize the session service with the database URL
session_service = DatabaseSessionService(db_url=DATABASE_URL)
memory_service = InMemoryMemoryService()

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
    city_normalized = city.lower().replace(" ", "") 

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

# Example tool usage (optional test)
print(get_weather("New York"))
print(get_weather("Paris"))

chatbot_agent = Agent(
    name="memory_chatbot",
    model=LiteLlm(model="gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY),
    description="Chatbot with persistent sessions",
    instruction="""You are an intelligent, expert and helpful ai assistant. Keep conversational context per session.
    When user asks queries only on weather exclusively, use the tool 'get_weather' to provide weather reports.
    For all other queries use your knowledge to provide answers(like even climate), 
    only for weather queries use the get_weather tool.
    Try to answer for almost all questions(excluding weather related queries) using your knowledge that you have.
    For example if user asks queries like "what is the climate of Paris" , you don't need to use the tool get_weather,
    you answer like "Paris has a temperate climate with mild winters and warm summers." using your knowledge.
    Use the tool get_weather only when needed 
    Note: Follow professional and ethical guidelines in all responses.
    """,
    tools = [get_weather]
    
)
root_agent = chatbot_agent

APP_NAME = "persistent_chatbot_app"
runner = Runner(agent=chatbot_agent, 
                app_name=APP_NAME, 
                session_service=session_service,
                memory_service=memory_service)

app = FastAPI()
origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3003", "http://127.0.0.1:3003","http:192.168.108.215:3000" ,"*", ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_query: str
    user_id: Optional[str]
    session_id: Optional[str]

class EnsureSessionRequest(BaseModel):
    user_id: str
    session_id: str

async def simple_ensure_session(app_name: str, user_id: str, session_id: str):
    """
    Simple session creation as fallback - based on reference code pattern
    """
    try:
        # Try to get existing session
        session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        if session is not None:
            return session
    except Exception as e:
        logger.debug(f"Session not found: {e}")
    
    # Create new session
    try:
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state={}
        )
        if session is None:
            raise RuntimeError("create_session returned None")
        return session
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise e

from typing import Optional, Dict, Any, List

def extract_messages_from_state(state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize stored session.state into a list of messages:
      [{ "sender": "user"|"bot", "text": "...", "role": "user|assistant|system", "idx": 0, "raw": {...} }, ...]
    Filtering rules:
      - Skip obvious tool / function_call messages (they remain in DB/state)
      - Prefer human-readable text from parts[*].text or parts[*].content or parts[*] str
      - Accept already-normalized {'sender','text'} entries
    """
    out: List[Dict[str, Any]] = []
    if not state:
        return out

    raw = state.get("messages") or state.get("history") or []
    if not isinstance(raw, list):
        return out

    for idx, m in enumerate(raw):
        try:
            if not isinstance(m, dict):
                # if it's a simple string, treat as user text
                if isinstance(m, str):
                    out.append({"sender": "user", "text": m, "role": "user", "idx": idx, "raw": m})
                continue

            # 1) If already normalized shape
            if "sender" in m and "text" in m:
                sender_val = m.get("sender")
                text_val = m.get("text")
                if sender_val and text_val:
                    out.append({"sender": sender_val, "text": text_val, "role": m.get("role"), "idx": idx, "raw": m})
                continue

            # Filtering out tool/function_call-like messages
            # common indicators: function_call in parts[*], explicit tool keys
            parts = m.get("parts") or []
            if isinstance(parts, list):
                has_function_call = any(isinstance(p, dict) and ("function_call" in p or "tool" in p or "tool_name" in p or "tool_output" in p) for p in parts)
                if has_function_call:
                    # skip tool events in history view
                    continue

            # 3) Mapping the role and sender
            role = m.get("role") or m.get("author") or m.get("type") or None
            if role:
                sender = "user" if role == "user" else "bot"
            else:
                sender = None

            # 4) Extract readable text from parts
            text = None
            if parts and isinstance(parts, list):
                # trying to find a part that is plain text
                for p in parts:
                    if isinstance(p, str):
                        text = p
                        break
                    if isinstance(p, dict):
                        # common keys where text may live
                        if "text" in p and isinstance(p["text"], str) and p["text"].strip():
                            text = p["text"]
                            break
                        if "content" in p and isinstance(p["content"], str) and p["content"].strip():
                            text = p["content"]
                            break
                        # skipping any function_call/tool objects deliberately
                        if "function_call" in p or "tool" in p or "tool_output" in p or "tool_name" in p:
                            text = None
                            break
            # As fallback, read top-level 'text' or 'message' keys if present
            if not text:
                for k in ("text", "message", "content"):
                    if k in m and isinstance(m[k], str) and m[k].strip():
                        text = m[k].strip()
                        break

            # Finally append if we got a sender and text
            if text and sender:
                out.append({"sender": sender, "text": text, "role": role, "idx": idx, "raw": m})
            # If no explicit role but we have text — assume user
            elif text:
                out.append({"sender": "user", "text": text, "role": role, "idx": idx, "raw": m})

        except Exception:
            continue

    return out

# adding context engineering techniques 
def approximate_token_count(messages: List[Dict[str, Any]]) -> int:
    """Very rough token count estimate.Each 4 chars ≈ 1 token (heuristic)."""
    total_chars = sum(len(m.get("text", "")) for m in messages)
    return total_chars // 4

def summarize_messages(messages: List[Dict[str, Any]], max_messages: int = 20) -> List[Dict[str, Any]]:
    """Summarize older messages to reduce context size.
    Keeps last `max_messages` intact and replaces earlier ones with a summary."""
    
    if len(messages) <= max_messages:
        return messages

    recent = messages[-max_messages:]
    old = messages[:-max_messages]
    summary_texts = [f"{m['sender']}: {m['text']}" for m in old]
    summary = {
        "sender": "system",
        "role": "system",
        "text": "Summary of earlier conversation:\n" + "\n".join(summary_texts[:50])  # limit for safety
    }
    return [summary] + recent

async def compress_and_store_history(session, session_service, token_limit: int = 4000):
    """
    Compress and store history if over token budget.
    """
    state = session.state or {}
    messages = extract_messages_from_state(state)

    token_count = approximate_token_count(messages)
    if token_count > token_limit:
        logger.info(f"History too large ({token_count} tokens) → summarizing")
        compressed = summarize_messages(messages)
        session.state["messages"] = compressed
        await session_service.update_session(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.session_id,
            state=session.state
        )

async def ensure_session_with_retries(app_name: str, user_id: str, session_id: str, 
                                     max_retries: int = 5, base_delay: float = 0.1):
    """
    Robust session creation with exponential backoff retry logic.
    This addresses timing issues between session creation and runner access.
    """
    if not user_id or not session_id:
        raise ValueError("user_id and session_id are required")

    logger.debug(f"ensure_session_with_retries: {user_id}/{session_id}")
    
    # get existing session
    try:
        session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        if session is not None:
            logger.debug("Session exists, returning existing session")
            return session
        else:
            logger.debug("get_session returned None")
    except Exception as e:
        logger.debug(f"Session doesn't exist: {e}")
    
    # create session with retries for database consistency
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"Creating session, attempt {attempt + 1}")
            
            # Create a session with empty initial state
            session = await session_service.create_session(
                app_name=app_name, 
                user_id=user_id, 
                session_id=session_id,
                state={}  # Start with empty state
            )
            
            if session is None:
                logger.warning(f"create_session returned None on attempt {attempt + 1}")
                raise RuntimeError("create_session returned None")
                
            logger.debug(f"Session created successfully: {type(session)} - {dir(session)}")
            
            # Wait a bit to ensure database consistency
            await asyncio.sleep(base_delay * (2 ** attempt))
            
            # Verifying the session can be retrieved
            try:
                verification_session = await session_service.get_session(
                    app_name=app_name, 
                    user_id=user_id, 
                    session_id=session_id
                )
                if verification_session is not None:
                    logger.debug("Session verified after creation")
                    return verification_session
                else:
                    logger.warning("Verification returned None, continuing to retry")
                    raise RuntimeError("Session verification returned None")
            except Exception as verify_exc:
                logger.warning(f"Session verification failed: {verify_exc}")
                raise verify_exc
            
        except Exception as create_exc:
            last_exception = create_exc
            logger.debug(f"Session creation attempt {attempt + 1} failed: {create_exc}")
            
            # If it failed due to already existing, try to get it
            try:
                existing_session = await session_service.get_session(
                    app_name=app_name, 
                    user_id=user_id, 
                    session_id=session_id
                )
                if existing_session is not None:
                    logger.debug("Found existing session after creation failure")
                    return existing_session
                else:
                    logger.debug("get_session returned None after creation failure")
            except Exception as get_exc:
                logger.debug(f"Failed to get session after creation failure: {get_exc}")
            
            if attempt == max_retries - 1:
                logger.error(f"All retry attempts exhausted. Last exception: {create_exc}")
                raise create_exc
            
            await asyncio.sleep(base_delay * (2 ** attempt))
    
    # exception handling after all retries
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"Failed to ensure session after {max_retries} attempts - unknown error")

async def run_agent_with_session_recovery(runner: Runner, user_id: str, session_id: str, 
                                        message: types.Content, max_attempts: int = 3):
    """
    Run agent with automatic session recovery on session not found errors.
    """
    for attempt in range(max_attempts):
        try:
            logger.info(f"Agent run attempt {attempt + 1} for session {session_id}")
            final_response = ""
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
                logger.debug(f"Event: {event}")
                if event.is_final_response():
                    final_response = event.content.parts[0].text
                    logger.info(f"Got final response: {final_response[:100]}...")
            return final_response
            
        except ValueError as ve:
            if "Session not found" in str(ve) and attempt < max_attempts - 1:
                logger.warning(f"Session not found on attempt {attempt + 1}, recreating session: {ve}")
                
                # Wait a bit before recreating to make database consistent
                await asyncio.sleep(0.2 * (attempt + 1))
                
                # Forcefully recreating a session
                try:
                    session = await ensure_session_with_retries(APP_NAME, user_id, session_id)
                    logger.info(f"Session recreated: {session}")
                    
                    # Additional wait for runner sync
                    await asyncio.sleep(0.5)
                    
                except Exception as recreate_exc:
                    logger.error(f"Failed to recreate session: {recreate_exc}")
                    if attempt == max_attempts - 1:
                        raise recreate_exc
                
                continue  
            else:
                # Not a session not found error or max attempts reached
                raise ve
        except Exception as e:
            logger.error(f"Unexpected error in agent run: {e}")
            raise e
    
    raise RuntimeError("Agent run failed after all recovery attempts")

# Endpoints 
@app.post("/sessions/ensure")
async def ensure_session_endpoint(req: EnsureSessionRequest):
    try:
        session = await ensure_session_with_retries(APP_NAME, req.user_id, req.session_id)
        if session is None:
            raise RuntimeError("ensure_session_with_retries returned None")
        
        # Use the session_id from the request since we know it exists
        return {"status": "ok", "session_exists": True, "session_id": req.session_id, "session_info": str(session)}
    except Exception as exc:
        logger.exception("ensure_session failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/history/{user_id}/{session_id}")
async def history(user_id: str, session_id: str):
    try:
        s = await session_service.get_session(APP_NAME, user_id, session_id)
        state = s.state or {}
        msgs = extract_messages_from_state(state)
        return {"messages": msgs}
    except Exception as exc:
        logger.debug("history: session missing or error: %s", exc)
        return {"messages": []}
@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    # Validating my input
    if not req.user_id or not req.session_id:
        raise HTTPException(status_code=400, detail="user_id and session_id are required")

    logger.info(f"Processing chat request for user_id={req.user_id}, session_id={req.session_id}")

    try:
        session = await ensure_session_with_retries(APP_NAME, req.user_id, req.session_id)
        if session is None:
            raise RuntimeError("Failed to create or retrieve session - session is None")

        #Compress the history if it is too large , adding context summarization in chat 
        await compress_and_store_history(session, session_service, token_limit=25000)

        # Build user message
        message = types.Content(role="user", parts=[types.Part(text=req.user_query)])

        # Run agent
        final_response = await run_agent_with_session_recovery(
            runner, req.user_id, req.session_id, message
        )

        return {"response": final_response}

    except Exception as exc:
        logger.exception("Chat endpoint error: %s", exc)
        tb = traceback.format_exc()
        if DEBUG:
            return {"error": str(exc), "traceback": tb}
        else:
            raise HTTPException(status_code=500, detail="Internal server error occurred")


# Testing my database connection
@app.get("/debug/db-test")
async def test_db_connection():
    try:
        # Try to create a test session to verify DB connectivity
        test_session = await session_service.create_session(
            app_name=f"{APP_NAME}_test",
            user_id="test_user",
            session_id=f"test_session_{asyncio.get_event_loop().time()}",
            state={"test": True}
        )
        return {"db_status": "connected", "test_session_id": test_session.session_id}
    except Exception as exc:
        return {"db_status": "error", "error": str(exc)}
