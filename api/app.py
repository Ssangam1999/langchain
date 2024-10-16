from pydoc_data.topics import topics
from venv import logger
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
import uvicorn
import  os
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Langchain Server",
    version ='1.0',
    description= 'A simple API Server'
)
load_dotenv()

llm = Ollama(model="llama3")
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")


class EssayRequest(BaseModel):
    topic:str
@app.post("/essay")
async def read_item(essay: EssayRequest):
    try:
        prompt_input = prompt1.format(topic = essay.topic)
        response = llm(prompt_input)
        return {"content": response}
    except Exception as e:
        logger.error(f"An error occured : {e}")


if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=8000)

