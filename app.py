import streamlit as st
import pandas as pd
from datetime import date
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import matplotlib.pyplot as plt
from prediksi import load_data
from prediksi import plot_raw_data
from prediksi import plot_raw_data
#from prediksi import dataframe_prediksi
from prediksi import dataframe_test_model
from prediksi import test_model
from prediksi import prediksi
from plotly import graph_objs as go


def coba():
    stocks = ('BBRI.JK', 'BBNI.JK','BTPN.JK','BMRI.JK')
    st.title('Data Aktual')
    selected_stock = st.selectbox('Pilih Kode Saham', stocks)
    data = load_data(selected_stock)
    st.subheader('Raw data')
    st.write(data.tail())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

page = st.sidebar.selectbox("Menu",("Home","Data Aktual", "Hasil Akurasi","Prediksi"))
if page == "Home":
    st.title('Aplikasi Prediksi Harga Saham Bank BUMN Menggunakan Model GRU (Gated Recurrent Unit)')
    image=Image.open("Stock_Market_Changes.png")
    st.image(image,use_column_width=True)
elif page == "Data Aktual":
    coba()
elif page == "Prediksi":
    st.title('Prediksi Harga Saham Bank BUMN Menggunakan Model Gated Recurrent Unit (GRU)')
    nama_model = "modelbtpn.h5"
    uploaded_file = st.file_uploader("Upload history harga saham 5 tahun", type=["csv"])
    if uploaded_file is not None:
        uploaded_file = pd.read_csv(uploaded_file)
        testing_tombol=st.button('Testing Model')
        st.title('Grafik Prediksi Harga 30 Hari Kedepan')
        prediksi(uploaded_file,nama_model)
elif page == "Hasil Akurasi":
    st.title('Hasil Akurasi Prediksi Harga Saham Bank BUMN Menggunakan Model Gated Recurrent Unit (GRU)')
    pilih=st.selectbox("Pilih Kode",('BBRI.JK', 'BBNI.JK','BTPN.JK','BMRI.JK'))
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        uploaded_file = pd.read_csv(uploaded_file)
        testing_tombol=st.button('Testing Model')
        if testing_tombol:
            dataframe_test_model(uploaded_file,pilih)

    
    

