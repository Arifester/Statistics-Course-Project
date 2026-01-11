from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import pandas as pd

@st.cache_resource
def train_model(df):
    """
    Melatih model Linear Regression.
    Menggunakan cache_resource karena model adalah objek berat.
    """
    X = df[['Days']]
    y = df['Close/Last']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=70)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mse": mse,
        "r2": r2,
        "test_score": model.score(X_test, y_test)
    }

    return model, metrics

def make_prediction(model, input_date, start_date):
    """
    Fungsi helper untuk melakukan prediksi satu titik data.
    """
    days_input = (input_date - start_date).days
    input_df = pd.DataFrame([[days_input]], columns=['Days'])
    prediction = model.predict(input_df)[0]
    return prediction
    