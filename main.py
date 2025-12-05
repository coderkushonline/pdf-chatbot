from conversation import GetPdfChatBot

from fastapi import FastAPI
from pydantic import BaseModel

class Model(BaseModel):
    filepath: str
    query: str
    descriptsion: str | None = None


app = FastAPI()

@app.post('/ask_ai')
def home(data: Model):
    try:
        chatbot = GetPdfChatBot(filepath=data.filepath)
        response = chatbot.qna(data.query)
        return {"content": response, "status": 200}
    
    except Exception as e:
        return {"content": "An error occured while connecting with the LLM. Please try again later", "status": 400}

