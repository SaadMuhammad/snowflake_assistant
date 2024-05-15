import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
from utils.logo import add_logo


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
