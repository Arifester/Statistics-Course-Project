import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import datetime

# Load data dan proses tanggal
df = pd.read_csv('https://raw.githubusercontent.com/Arifester/Datasets/main/gold%20prices.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Hitung jumlah hari sejak tanggal pertama
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Pilih fitur dan target
X = df[['Days']]
y = df['Close/Last']

# Bagi data untuk pelatihan dan pengujian (optional, kamu juga bisa pakai semua data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=70)

# Buat dan latih model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi model (optional)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error (MSE) pada data test: {mse:.2f}')

# Judul dan gambar aplikasi
st.title('Prediksi Harga Emas')
st.image('https://www.mentarimulia.co.id/wp-content/uploads/2022/07/emas-5.jpg')

# Input bulan dan tahun dari user
bulan = st.slider("Pilih Bulan (1-12)", 1, 12, 1)
tahun = st.number_input("Masukkan Tahun (contoh: 2024)", value=2024)

# Buat objek tanggal dari input user
input_date = pd.Timestamp(year=int(tahun), month=int(bulan), day=1)

# Validasi input tanggal
if input_date < df['Date'].min():
    st.error(f"Input tanggal tidak valid. Tanggal harus setelah {df['Date'].min().date()}.")
else:
    days_input = (input_date - df['Date'].min()).days

    # Bungkus input menjadi DataFrame dengan nama kolom yang sama dengan data training
    input_df = pd.DataFrame([[days_input]], columns=['Days'])

    # Prediksi harga emas
    predicted_price = model.predict(input_df)

    # Tampilkan hasil prediksi
    st.write(f'Prediksi harga emas untuk bulan {bulan} tahun {tahun}: ${predicted_price[0]:.2f}')
