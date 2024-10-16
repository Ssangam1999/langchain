from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
# Fetch API keys
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# Langchain prompt setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)
# Streamlit framework
st.title("Langchain Demo with llama3")
input_text = st.text_input("Search the topic you want")

# Initialize OpenAI LLM with API key
llm = Ollama(model = 'llama3')
# Output parser
output_parser = StrOutputParser()

# Chain of prompt, LLM, and parser
chain = prompt | llm | output_parser

# Execute when input is provided
if input_text:
    try:
        result = chain.invoke({'question': input_text})
        st.write(result)
    except Exception as e:
        st.error(f"Error: {str(e)}")

