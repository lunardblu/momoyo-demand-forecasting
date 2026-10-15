
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Demand Forecasting Minuman",
    page_icon="🥤",
    layout="wide"
)

st.title("Demand Forecasting Penjualan Minuman")
st.write(
    "Aplikasi untuk menganalisis penjualan dan memprediksi "
    "permintaan minuman berdasarkan data historis."
)

st.info(
    "Data yang digunakan merupakan data simulasi untuk keperluan pembelajaran, "
    "bukan data penjualan resmi Momoyo."
)

# Membuat data simulasi selama 90 hari
np.random.seed(42)

tanggal = pd.date_range(
    start="2026-01-01",
    periods=90,
    freq="D"
)

produk_list = ["Milk Tea", "Brown Sugar", "Matcha"]

data_produk = []

for produk in produk_list:
    dasar = {
        "Milk Tea": 45,
        "Brown Sugar": 35,
        "Matcha": 25
    }[produk]

    tren = np.linspace(0, 10, 90)
    akhir_pekan = np.array([
        10 if t.dayofweek >= 5 else 0
        for t in tanggal
    ])

    penjualan = (
        dasar + tren + akhir_pekan
        + np.random.normal(0, 5, 90)
    ).round().astype(int)

    penjualan = np.maximum(penjualan, 0)

    for t, jumlah in zip(tanggal, penjualan):
        data_produk.append({
            "Tanggal": t,
            "Produk": produk,
            "Penjualan": jumlah
        })

df = pd.DataFrame(data_produk)

# Sidebar
st.sidebar.header("Pengaturan Forecasting")

produk = st.sidebar.selectbox(
    "Pilih produk minuman",
    produk_list
)

metode = st.sidebar.selectbox(
    "Metode forecasting",
    ["Moving Average", "Exponential Smoothing"]
)

periode = st.sidebar.slider(
    "Periode rata-rata (hari)",
    min_value=3,
    max_value=14,
    value=7
)

horizon = st.sidebar.slider(
    "Prediksi ke depan (hari)",
    min_value=1,
    max_value=14,
    value=7
)

# Memfilter data produk
data = (
    df[df["Produk"] == produk]
    .sort_values("Tanggal")
    .reset_index(drop=True)
)

nilai = data["Penjualan"].astype(float)

# Menghitung prediksi historis untuk evaluasi
if metode == "Moving Average":
    prediksi_lalu = nilai.shift(1).rolling(
        window=periode
    ).mean()
else:
    prediksi_lalu = (
        nilai.shift(1).ewm(
            alpha=2 / (periode + 1),
            adjust=False
        ).mean()
    )

valid = prediksi_lalu.notna()

if valid.any():
    mae = np.mean(
        np.abs(
            nilai[valid] - prediksi_lalu[valid]
        )
    )
else:
    mae = 0

# Membuat prediksi periode berikutnya
riwayat = list(nilai)

if metode == "Moving Average":
    ramalan = []

    for _ in range(horizon):
        hasil = float(np.mean(riwayat[-periode:]))
        ramalan.append(hasil)
        riwayat.append(hasil)

else:
    ramalan = []
    alpha = 2 / (periode + 1)
    level = float(nilai.iloc[0])

    for aktual in nilai.iloc[1:]:
        level = alpha * float(aktual) + (1 - alpha) * level

    for _ in range(horizon):
        ramalan.append(level)

tanggal_ramalan = pd.date_range(
    start=data["Tanggal"].iloc[-1] + pd.Timedelta(days=1),
    periods=horizon,
    freq="D"
)

# Ringkasan
kolom1, kolom2, kolom3 = st.columns(3)

kolom1.metric(
    "Total Penjualan",
    f"{int(data['Penjualan'].sum()):,} unit"
)

kolom2.metric(
    "Rata-rata Harian",
    f"{data['Penjualan'].mean():.1f} unit"
)

kolom3.metric(
    "MAE",
    f"{mae:.2f} unit"
)

st.subheader(f"Riwayat Penjualan {produk}")
st.dataframe(data, use_container_width=True)

# Grafik historis dan prediksi
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    data["Tanggal"],
    data["Penjualan"],
    label="Penjualan Aktual"
)

ax.plot(
    tanggal_ramalan,
    ramalan,
    marker="o",
    linestyle="--",
    label="Hasil Forecasting"
)

ax.axvline(
    data["Tanggal"].iloc[-1],
    linestyle=":",
    label="Awal Periode Prediksi"
)

ax.set_xlabel("Tanggal")
ax.set_ylabel("Jumlah Terjual (unit)")
ax.set_title(f"Forecasting Penjualan {produk}")
ax.legend()
ax.grid(alpha=0.3)

fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

# Tabel hasil prediksi
hasil_prediksi = pd.DataFrame({
    "Tanggal": tanggal_ramalan,
    "Produk": produk,
    "Prediksi Penjualan": [
        round(x, 2) for x in ramalan
    ]
})

st.subheader("Hasil Prediksi")
st.dataframe(
    hasil_prediksi,
    use_container_width=True
)

st.download_button(
    "Download Hasil Prediksi CSV",
    data=hasil_prediksi.to_csv(index=False).encode("utf-8"),
    file_name="hasil_forecasting.csv",
    mime="text/csv"
)

st.caption(
    "Catatan: hasil prediksi merupakan estimasi berdasarkan data simulasi "
    "dan bukan jaminan penjualan aktual."
)
