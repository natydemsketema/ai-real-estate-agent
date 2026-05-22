from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

load_dotenv()
app = FastAPI(title="AI Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials= True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
chat_histories={}

class ChatRequest(BaseModel):
    message:str
    session_id: str="default"

@app.get("/")
async def root():
    return{"status" :" Ai Cahtboot is alive"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.session_id not in chat_histories:
        chat_histories[request.session_id]=[
          SystemMessage(content=(
                "You are an expert AI Real Estate Agent for Zembil Real Estate. "
                "You must ONLY talk about real estate development, properties, houses, and land. "
                "If the user asks about anything else (like coding, cooking, or general jokes), "
                "politely refuse and guide them back to real estate options."

            ))
        ]
    chat_histories[request.session_id].append(HumanMessage(content=request.message))
    response = llm.invoke(chat_histories[request.session_id])
    chat_histories[request.session_id].append(response)
    return{
        "response": response.content,
        "session_id" : request.session_id
    }
if __name__=="__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

    