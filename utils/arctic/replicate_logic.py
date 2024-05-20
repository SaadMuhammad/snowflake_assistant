import replicate
import streamlit as st
import os

os.environ["REPLICATE_API_TOKEN"] = "".join(elem for elem in st.secrets["REPLICATE_API_TOKEN"])
api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
