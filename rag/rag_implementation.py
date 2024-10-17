from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
import bs4
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()
os.environ['USER_AGENT'] = os.getenv('LANGCHAIN_API_KEY')
loader = TextLoader("speech.txt")
text_documents=loader.load()
llm = OllamaLLM(model="llama3")
loader = WebBaseLoader(web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=("post-title","post_content","post_title")

                       )))

text_documents = loader.load()
loader = PyPDFLoader('attention.pdf')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents = text_splitter.split_documents(docs)
# db = Chroma.from_documents(documents[:2],OllamaEmbeddings(model='llama3'))
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(documents[:15], OllamaEmbeddings(model='llama3'))
query = "Who are the authors of attention is all you need?"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)
