from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# Ensure your API key is loaded
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please check your environment variables.")

# Langchain prompt setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Search the topic you want")

# Initialize OpenAI LLM with API key
llm = ChatOpenAI(api_key=OPENAI_API_KEY,model = 'gpt-4o-mini')
# llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY)

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
