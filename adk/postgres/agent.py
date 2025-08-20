import os
import asyncio
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from google.adk.agents import Agent
from google.adk.sessions import DatabaseSessionService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

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

# Example tool usage
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
