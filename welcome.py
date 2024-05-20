import streamlit as st
import pandas as pd
from PIL import Image
import os
import base64
from io import BytesIO
from utils.arctic.replicate_logic import api

st.set_page_config(page_title="SnowFlake Assitant", page_icon="❄️🔍", layout="wide")
                   
            
st.markdown(
    
    f"""
    <style>
        [data-testid="stSidebarNav"] {{
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
    #image = Image.open(r"SnowFlake_Assistant.PNG") 
    image = Image.open(r"images/SnowFlake_Assistant.jpg") 
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

input = {
    "prompt": f"{user_question}",
    "temperature": 0.2
}

output = api.run(
    "snowflake/snowflake-arctic-instruct",
    input=input
)

# Use the selected predefined question if the user didn't enter a question  
question = user_question or selected_question  
      
if question:  
    # Append user's question to messages  
    st.session_state.messages.append({"role": "user", "content": question})  
    
    # Append assistant's response to messages  
    response = st.write_stream(output)  
    st.session_state.messages.append({"role": "assistant", "content": response})
          
