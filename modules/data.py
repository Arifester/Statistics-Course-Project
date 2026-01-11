import pandas as pd
import streamlit as st

@st.cache_data
def load_and_clean_data(url):
    """
    Load data dari URL dan lakukan preprocessing dasar.
    """
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Urutkan berdasarkan tanggal agar plot rapi
        df = df.sort_values('Date')
        
        # Hitung jumlah hari sejak tanggal pertama (untuk fitur regresi)
        start_date = df['Date'].min()
        df['Days'] = (df['Date'] - start_date).dt.days
        
        return df, start_date
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
        