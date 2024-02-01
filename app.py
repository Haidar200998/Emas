import streamlit as st
import numpy as np
import pickle

# Memuat model yang telah disimpan
with open('DecisionTree_best_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)
with open('RandomForest_best_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
with open('AdaBoost_best_model.pkl', 'rb') as file:
    ada_model = pickle.load(file)

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Emas", page_icon=":moneybag:")

# Membuat judul dan deskripsi aplikasi
st.title("Prediksi Harga Emas")
st.write("Prediksi harga emas menggunakan model Decision Tree, Random Forest, atau AdaBoost.")

# Membuat input untuk pengguna
ihsg = st.number_input('Masukkan Nilai IHSG', format="%.2f")
kurs_jual = st.number_input('Masukkan Kurs Jual', format="%.2f")
# Pengguna hanya perlu memasukkan angka seperti 2.9 (yang akan diinterpretasikan sebagai 2.9%)
data_inflasi = st.number_input('Masukkan Data Inflasi (dalam persen, contoh: masukkan 2.9 untuk 2.9%)', format="%.2f")

# Memungkinkan pengguna memilih model yang akan digunakan untuk prediksi
model_option = st.selectbox("Pilih Model untuk Prediksi:", ['Decision Tree', 'Random Forest', 'AdaBoost'])

# Menentukan model berdasarkan pilihan pengguna
if model_option == 'Decision Tree':
    model = dt_model
elif model_option == 'Random Forest':
    model = rf_model
else:  # 'AdaBoost'
    model = ada_model

# Tombol untuk melakukan prediksi
predict_btn = st.button("Prediksi Harga")

# Prediksi dan menampilkan hasil
if predict_btn:
    # Mengonversi input inflasi dari persentase ke desimal secara otomatis
    inflasi_desimal = data_inflasi / 100
    # Membuat prediksi
    inputs = np.array([[ihsg, kurs_jual, inflasi_desimal]])
    predicted_price = model.predict(inputs)[0]
    
    # Menampilkan hasil prediksi
    st.write("")
    st.subheader(f"Harga Emas yang Diprediksi: Rp {predicted_price:,.2f} menggunakan {model_option}")
