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

from chatbot.localama import prompt

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
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}""")
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm,prompt)
""""
A retriever is an interface that returns documents given an unstructured query. 
It is more general than a vector store. A retriever does not need to be able to store documents, 
only to return (or retrieve) them.
Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well.
https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/
"""
retriever =db.as_retriever()
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever,document_chain)
response = retrieval_chain.invoke({"input":"An attention function can be described as mapping query"})
print("Answer from retrieval is")
print(response['answer'])