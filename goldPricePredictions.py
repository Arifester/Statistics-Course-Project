import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import datetime

df = pd.read_csv('https://raw.githubusercontent.com/Arifester/Datasets/main/gold%20prices.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Menghitung jumlah hari sejak tanggal pertama dalam dataset
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Memilih fitur dan target
X = df[['Days']].values
y = df['Close/Last'].values

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=70)

# Membuat model regresi linier
model = LinearRegression()

# Melatih model
model.fit(X_train, y_train)

# Membuat prediksi
y_pred = model.predict(X_test)

# Menghitung Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)


# Konversi kolom 'Date' menjadi tipe datetime
df['Date'] = pd.to_datetime(df['Date'])

# Menghitung jumlah hari sejak tanggal pertama dalam dataset
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Membuat model regresi linier
model = LinearRegression()
X = df[['Days']]
y = df['Close/Last']
model.fit(X, y)

# Judul aplikasi
st.title('Prediksi Harga Emas')

# Input bulan dan tahun
bulan = st.slider("Pilih Bulan (1-12)", 1, 12, 1)
tahun = st.number_input("Masukkan Tahun (contoh: 2024)", value=2024)  # Default value untuk tahun

# Konversi bulan dan tahun menjadi bilangan bulat
bulan = int(bulan)
tahun = int(tahun)

# Membuat objek timestamp berdasarkan input bulan dan tahun
input_date = pd.Timestamp(year=tahun, month=bulan, day=1)

# Validasi input_date
if input_date < df['Date'].min():
    st.error(f"Input tanggal tidak valid. Tanggal harus setelah {df['Date'].min().date()}.")
else:
    # Menghitung jumlah hari berdasarkan bulan dan tahun yang diinput
    days_input = (input_date - df['Date'].min()).days

    # Membuat prediksi harga emas untuk bulan dan tahun yang diinput
    predicted_price = model.predict([[days_input]])

    # Menampilkan hasil prediksi
    st.write(f'Prediksi harga emas untuk bulan {bulan} tahun {tahun}: ${predicted_price[0]:.2f}')
