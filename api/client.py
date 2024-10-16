import requests
import streamlit as st




def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay",
        json={'topic': input_text}
    )
    return response.json()['content']


    # # Check if the response is successful and contains the expected data
    # if response.status_code == 200:
    #     json_response = response.json()
    #     # Access the 'content' key based on the FastAPI response structure
    #     return json_response.get('content', "No content returned.")
    # else:
    #     st.error(f"Error: Received status code {response.status_code}")
    #     st.write(response.text)  # For debugging purposes

    ## streamlit framework

st.title('Langchain Demo With LLAMA2 API')
input_text1=st.text_input("Write a essay on")


if input_text1:
    st.write(get_ollama_response(input_text1))