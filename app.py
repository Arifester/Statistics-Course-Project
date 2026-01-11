import streamlit as st
import pandas as pd
import datetime

# Import modul buatan sendiri
from modules import data, model, visual

# 1. Konfigurasi Halaman (Harus di baris pertama)
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="💰",
    layout="wide"
)

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi Prediksi")
        st.write("Atur parameter waktu untuk memprediksi harga emas di masa depan.")
        
        # Input User
        current_year = datetime.datetime.now().year
        bulan = st.selectbox("Pilih Bulan", range(1, 13), index=0)
        tahun = st.number_input("Masukkan Tahun", min_value=2000, max_value=2050, value=current_year)
        
        st.markdown("---")
        st.caption("Proyek Mata Kuliah Statistika")

    # --- MAIN CONTENT ---
    st.title("💰 Analisis & Prediksi Harga Emas")
    st.markdown("Aplikasi ini menggunakan **Linear Regression** untuk menganalisis tren historis dan memprediksi harga emas di masa depan.")

    # 2. Load Data
    DATA_URL = 'https://raw.githubusercontent.com/Arifester/Datasets/main/gold%20prices.csv'
    df, start_date = data.load_and_clean_data(DATA_URL)

    if df is not None:
        # 3. Train Model
        reg_model, metrics = model.train_model(df)

        # 4. Tampilkan Metrik Kinerja Model (Baris Atas)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data Points", len(df))
        with col2:
            st.metric("Model Accuracy (R²)", f"{metrics['r2']:.2%}")
        with col3:
            latest_price = df['Close/Last'].iloc[-1]
            st.metric("Harga Terakhir (Dataset)", f"${latest_price:.2f}")

        # 5. Proses Prediksi User
        input_date = pd.Timestamp(year=int(tahun), month=int(bulan), day=1)
        
        if input_date < start_date:
            st.error(f"⚠️ Tanggal prediksi harus setelah {start_date.date()}")
        else:
            predicted_price = model.make_prediction(reg_model, input_date, start_date)
            
            # Tampilkan Hasil Prediksi dengan Highlight
            st.success(f"### 🎯 Prediksi Harga: ${predicted_price:.2f}")
            
            # 6. Visualisasi
            st.subheader("Grafik Pergerakan Harga")
            chart = visual.plot_gold_trend(df, input_date, predicted_price)
            st.plotly_chart(chart, use_container_width=True)

            # Penjelasan Statistik (Opsional, untuk nilai tambah mata kuliah)
            with st.expander("ℹ️ Penjelasan Statistik Model"):
                st.write(f"""
                Model menggunakan algoritma **Linear Regression**.
                - **MSE (Mean Squared Error):** {metrics['mse']:.2f} (Semakin kecil semakin baik)
                - **R² Score:** {metrics['r2']:.4f} (Menunjukkan seberapa baik garis regresi cocok dengan data)
                
                Rumus dasar: $y = mx + c$, dimana $x$ adalah jumlah hari sejak {start_date.date()}.
                """)

if __name__ == "__main__":
    main()
    