import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
from io import BytesIO
from utils.logo import add_logo
from utils.arctic.auth import generate_embeddings_arctic, faiss_search
from web_test.snwflk_app.utils.arctic.arctic1 import generate_ai_response


st.set_page_config(page_title="SnowFlake Assitant", page_icon="❄️🔍", layout="wide")
                   

logo_data = add_logo()

st.markdown(
    f"""
    <style>
        [data-testid="stSidebarNav"] {{
            background-image: url(data:image/png;base64,{logo_data});
            background-repeat: no-repeat;
            padding-top: 120px;
            background-position: 20px 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

if 'messages' not in st.session_state:  
    st.session_state.messages = []  

col1a, col2a, col3a = st.columns([2,3,2])

with col2a:
    image = Image.open(r"SnowFlake_Assistant.PNG")  
    st.image(image, use_column_width=True)

col01, col02, col03 = st.columns([1,4,1])

with col02:
    st.title('Your SnowFlake Assitant')

# Predefined questions  
predefined_questions = ["snowflake ques1", "snowflake ques2", "snowflake ques3"]  
  
# Create a row of columns for the predefined questions  
columns = st.columns(len(predefined_questions))  
  
selected_question = None  
for i, question in enumerate(predefined_questions):  
    if columns[i].button(question):  
        selected_question = question  
        break 

# Display chat history  
for message in st.session_state.messages:  
    with st.chat_message(message["role"]):  
        st.markdown(message["content"])

 # Ask the user to enter a question  
user_question = st.text_input('Please enter your question here:') 

# Use the selected predefined question if the user didn't enter a question  
question = user_question or selected_question  
      
if question:  
    # Append user's question to messages  
    st.session_state.messages.append({"role": "user", "content": question})  
    #query = generate_embeddings(str(question))
    context = faiss_search(question) #faiss search title and give indices call another df and create temp faiss index
    resp = generate_ai_response(context, question)

    # Append assistant's response to messages  
    response = st.write_stream(resp)
    st.session_state.messages.append({"role": "assistant", "content": response})
          
