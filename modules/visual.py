import plotly.graph_objects as go
import plotly.express as px

def plot_gold_trend(df, input_date=None, predicted_price=None):
    """
    Membuat chart interaktif harga emas + titik prediksi user.
    """
    fig = go.Figure()

    # Plot Data Historis
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close/Last'],
        mode='lines',
        name='Harga Historis',
        line=dict(color='#FFD700') # Warna emas
    ))

    # Jika ada prediksi user, tambahkan marker
    if input_date and predicted_price:
        fig.add_trace(go.Scatter(
            x=[input_date],
            y=[predicted_price],
            mode='markers+text',
            name='Prediksi Kamu',
            marker=dict(color='red', size=12, symbol='star'),
            text=[f"${predicted_price:.2f}"],
            textposition="top center"
        ))

    fig.update_layout(
        title='Tren Harga Emas & Prediksi',
        xaxis_title='Tahun',
        yaxis_title='Harga (USD)',
        template='plotly_dark', # Tema gelap agar elegan
        hovermode="x unified"
    )

    return fig
    