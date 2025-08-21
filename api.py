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

# -------- Config & logging --------
load_dotenv()
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chat-api")

# -------- Services & Agent --------
session_service = DatabaseSessionService(db_url=DATABASE_URL)

chatbot_agent = Agent(
    name="memory_chatbot",
    model=LiteLlm(model="gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY),
    description="Chatbot with persistent sessions",
    instruction="You are a helpful assistant. Keep conversational context per session."
)

APP_NAME = "persistent_chatbot_app"
runner = Runner(agent=chatbot_agent, app_name=APP_NAME, session_service=session_service)

# -------- FastAPI & CORS --------
app = FastAPI()
origins = ["http://localhost:3003", "http://127.0.0.1:3003","http://localhost:3000", "http://127.0.0.1:3000", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Request models --------
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
def _extract_messages_from_state(state: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize stored session.state into a list of {sender, text} messages."""
    messages = []
    if not state:
        return messages
    raw = state.get("messages") or state.get("history") or []
    if not isinstance(raw, list):
        return messages
    for m in raw:
        sender = None
        text = None
        if isinstance(m, dict):
            if "sender" in m and "text" in m:
                sender, text = m.get("sender"), m.get("text")
            else:
                role = m.get("role") or m.get("author")
                parts = m.get("parts") or []
                if role:
                    sender = "user" if role == "user" else "bot"
                if parts and isinstance(parts, list):
                    p0 = parts[0]
                    if isinstance(p0, dict):
                        text = p0.get("text") or p0.get("content")
                    elif isinstance(p0, str):
                        text = p0
        if sender and text:
            messages.append({"sender": sender, "text": text})
    return messages

async def ensure_session_with_retries(app_name: str, user_id: str, session_id: str, 
                                     max_retries: int = 5, base_delay: float = 0.1):
    """
    Robust session creation with exponential backoff retry logic.
    This addresses timing issues between session creation and runner access.
    """
    if not user_id or not session_id:
        raise ValueError("user_id and session_id are required")

    logger.debug(f"ensure_session_with_retries: {user_id}/{session_id}")
    
    # First, try to get existing session
    try:
        session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
        if session is not None:
            logger.debug("Session exists, returning existing session")
            return session
        else:
            logger.debug("get_session returned None")
    except Exception as e:
        logger.debug(f"Session doesn't exist: {e}")
    
    # Try to create session with retries for database consistency
    last_exception = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"Creating session, attempt {attempt + 1}")
            
            # Create session with empty initial state
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
                # Don't return here, continue to retry logic
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
    
    # This should not be reached, but just in case
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
                
                # Wait a bit before recreating
                await asyncio.sleep(0.2 * (attempt + 1))
                
                # Force recreate session
                try:
                    session = await ensure_session_with_retries(APP_NAME, user_id, session_id)
                    logger.info(f"Session recreated: {session}")
                    
                    # Additional wait for runner sync
                    await asyncio.sleep(0.5)
                    
                except Exception as recreate_exc:
                    logger.error(f"Failed to recreate session: {recreate_exc}")
                    if attempt == max_attempts - 1:
                        raise recreate_exc
                
                continue  # Try again
            else:
                # Not a session not found error or max attempts reached
                raise ve
        except Exception as e:
            logger.error(f"Unexpected error in agent run: {e}")
            raise e
    
    raise RuntimeError("Agent run failed after all recovery attempts")

# -------- Endpoints --------
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
        msgs = _extract_messages_from_state(state)
        return {"messages": msgs}
    except Exception as exc:
        logger.debug("history: session missing or error: %s", exc)
        return {"messages": []}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    # Validate input
    if not req.user_id or not req.session_id:
        raise HTTPException(status_code=400, detail="user_id and session_id are required")

    logger.info(f"Processing chat request for user_id={req.user_id}, session_id={req.session_id}")

    try:
        # Ensure session exists with robust retry logic
        session = await ensure_session_with_retries(APP_NAME, req.user_id, req.session_id)
        
        if session is None:
            raise RuntimeError("Failed to create or retrieve session - session is None")
            
        logger.info(f"Session ensured for session_id: {req.session_id} - Session object: {type(session)}")

        # Build user message
        message = types.Content(role="user", parts=[types.Part(text=req.user_query)])

        # Run agent with automatic session recovery
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

# Optional debug endpoint to inspect session existence and raw state
@app.get("/debug/session/{user_id}/{session_id}")
async def debug_session(user_id: str, session_id: str):
    try:
        s = await session_service.get_session(APP_NAME, user_id=user_id, session_id=session_id)
        if s is None:
            return {"exists": False, "error": "get_session returned None"}
        
        # Get session attributes safely
        session_attrs = {attr: getattr(s, attr, "N/A") for attr in dir(s) if not attr.startswith('_')}
        
        return {
            "exists": True, 
            "state": getattr(s, 'state', None), 
            "session_id_param": session_id,
            "session_type": str(type(s)),
            "session_attributes": session_attrs
        }
    except Exception as exc:
        return {"exists": False, "error": str(exc)}

# Alternative endpoint using simpler session creation
@app.post("/chat-simple")
async def chat_simple_endpoint(req: ChatRequest, request: Request):
    # Validate input
    if not req.user_id or not req.session_id:
        raise HTTPException(status_code=400, detail="user_id and session_id are required")

    logger.info(f"Processing simple chat request for user_id={req.user_id}, session_id={req.session_id}")

    try:
        # Use simpler session creation
        session = await simple_ensure_session(APP_NAME, req.user_id, req.session_id)
        
        if session is None:
            raise RuntimeError("Failed to create or retrieve session - session is None")
            
        logger.info(f"Session created/found for session_id: {req.session_id} - Session object: {type(session)}")

        # Build user message
        message = types.Content(role="user", parts=[types.Part(text=req.user_query)])

        # Try direct runner call first
        try:
            final_response = ""
            async for event in runner.run_async(user_id=req.user_id, session_id=req.session_id, new_message=message):
                if event.is_final_response():
                    final_response = event.content.parts[0].text
            
            return {"response": final_response}
            
        except ValueError as ve:
            if "Session not found" in str(ve):
                # Wait and try once more
                await asyncio.sleep(0.5)
                final_response = ""
                async for event in runner.run_async(user_id=req.user_id, session_id=req.session_id, new_message=message):
                    if event.is_final_response():
                        final_response = event.content.parts[0].text
                return {"response": final_response}
            else:
                raise ve

    except Exception as exc:
        logger.exception("Simple chat endpoint error: %s", exc)
        tb = traceback.format_exc()
        
        if DEBUG:
            return {"error": str(exc), "traceback": tb}
        else:
            raise HTTPException(status_code=500, detail=str(exc))
@app.get("/health")
async def health_check():
    return {"status": "healthy", "app_name": APP_NAME}

# Test database connection
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