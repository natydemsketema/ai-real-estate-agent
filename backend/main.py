import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Configure Production-Grade Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify critical environment variables at startup
if not os.getenv("GOOGLE_API_KEY"):
    logger.critical("Initialization Failed: GOOGLE_API_KEY environment variable is missing!")
    # In production, you often want to crash early if a critical secret is missing
    # raise ValueError("GOOGLE_API_KEY is required to run this application.")

app = FastAPI(
    title="Zembil Real Estate AI API", 
    version="1.0.0",
    description="Production-ready backend endpoint handling scoped real estate AI conversations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM (It automatically picks up GOOGLE_API_KEY from environment)
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7
    )
    logger.info("Successfully initialized ChatGoogleGenerativeAI instance.")
except Exception as init_err:
    logger.critical(f"Failed to initialize LLM client setup: {str(init_err)}")
    raise init_err

# In-memory history cache (Note: Swap with Redis/Database for production horizontal scaling)
chat_histories = {}

# 2. Strict Input Validation with Pydantic
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The textual message sent by the user.")
    session_id: str = Field(default="default", description="Unique identifier tracking user chat sessions.")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Do you have any commercial buildings available in Mekelle?",
                "session_id": "session_natnael_123"
            }
        }

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"status": "AI Chatbot is operational and active."}

# 3. Secure and Observable Chat Endpoint
@app.post("/chat", status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message.strip()
    
    logger.info(f"Incoming chat request received for Session ID: '{session_id}'")

    # Initialize session history sequence with guardrails if not present
    if session_id not in chat_histories:
        logger.info(f"Session '{session_id}' not found in cache. Initializing new conversation stream.")
        chat_histories[session_id] = [
            SystemMessage(content=(
                "You are an expert AI Real Estate Agent for Zembil Real Estate. "
                "You must ONLY talk about real estate development, properties, houses, and land. "
                "If the user asks about anything else (like coding, cooking, or general jokes), "
                "politely refuse and guide them back to real estate options."
            ))
        ]

    # Append user input safely
    chat_histories[session_id].append(HumanMessage(content=user_message))

    # 4. Graceful Exception Handling for External API Invocations
    try:
        logger.info(f"Invoking Gemini model pipeline for Session: '{session_id}'")
        
        # Call the model
        response = llm.invoke(chat_histories[session_id])
        
        # Append AI response to maintain memory state continuity
        chat_histories[session_id].append(response)
        
        logger.info(f"Successfully processed LLM generation sequence for Session: '{session_id}'")
        
        return {
            "response": response.content,
            "session_id": session_id
        }

    except KeyError as key_err:
        logger.error(f"State handling failure for session '{session_id}': {str(key_err)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal state tracking error occurred while processing your request."
        )
        
    except Exception as api_err:
        # Catch network timeouts, rate limit overages, or invalid credentials safely
        logger.error(f"Upstream LLM Provider Failure on session '{session_id}': {str(api_err)}")
        
        # Remove the un-replied human message so the history queue doesn't desynchronize
        if chat_histories[session_id] and isinstance(chat_histories[session_id][-1], HumanMessage):
            chat_histories[session_id].pop()
            
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI Engine is temporarily unavailable. Please try your request again shortly."
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)