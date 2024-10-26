import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Chatgroq with llama3")
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name='llama3-8b-8192')
prompt = ChatPromptTemplate.from_template(
    """
    There is a story in provided document. you have to go through it and found out examples that can relate to machine
    learning. few examples are given below:
    example 1. Santiago is like the machine learning model that is trying to reach an end goal or make sense of the world. His journey to find the treasure is akin to the learning process where
     the model is trained to map inputs (experiences and lessons) to outputs (understanding his personal legend).
     example 2. Throughout his journey, Santiago is guided by mentors like the alchemist, who give him wisdom and teach him how to interpret signs and omens. This is analogous to labeled data in supervised learning, where the input comes with the correct answer or guidance (the labels). The mentors help Santiago learn how to navigate his journey,
     just as labeled data helps a model learn how to predict or classify new inputs.
     example 3. Santiago makes mistakes, faces challenges, and reflects on his actions. In supervised learning, when a model makes a wrong prediction, it receives feedback and adjusts its internal parameters to minimize future errors, 
     much like Santiago learns from his experiences and grows wiser.
     example 4. Santiago's treasure at the end of the story symbolizes the model's ability to make accurate predictions after having learned from the data (mentors and experiences). It’s the culmination of the learning process, similar to how a well-trained
      model achieves a high level of accuracy after absorbing knowledge from the data.
      Answer the questions based on the provided context only.
      Please provide most accurate response based on the query.
      <context>
      {context}
      <context>
      Questions: {input}
    
    """ )


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model='llama3')
        st.session_state.loader=PyPDFLoader("alchemist.pdf") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector  embeddings


prompt1 = st.text_input("Enter your question related to machine learning")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")


if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response=retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------")




