import streamlit as st
import pandas as pd
from PIL import Image
import math
import numpy as np
from datetime import date
from plotly import graph_objs as go
import yfinance as yf

import matplotlib.pyplot as plt


st.title('Stock Forecast App Using GRU Model')

#image=Image.open("C:/Users/win10/webapp/Stock_Market_Changes.png")
#st.image(image,use_column_width=True)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('BBRI.JK', 'BBNI.JK')
st.sidebar.title('Data Aktual')
selected_stock = st.sidebar.selectbox('Choose stock market dataset ', stocks)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

    
#data_load_state = st.sidebar.text('Loading data...')
data = load_data(selected_stock)

st.subheader('Raw data')
st.write(data.tail())