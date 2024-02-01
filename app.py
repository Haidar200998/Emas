import streamlit as st
import numpy as np
import pickle

# Memuat model yang telah disimpan
with open('/content/drive/MyDrive/Skripsi 2/DecisionTree_best_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)
with open('/content/drive/MyDrive/Skripsi 2/RandomForest_best_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
with open('/content/drive/MyDrive/Skripsi 2/AdaBoost_best_model.pkl', 'rb') as file:
    ada_model = pickle.load(file)

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(page_title="Gold Price Prediction", page_icon=":moneybag:")

# Membuat judul dan deskripsi aplikasi
st.title("Gold Price Prediction")
st.write("Predict the gold price using Decision Tree, Random Forest, or AdaBoost models.")

# Membuat input untuk pengguna
ihsg = st.number_input('Enter IHSG Value', format="%.2f")
kurs_jual = st.number_input('Enter Exchange Rate (Kurs Jual)', format="%.2f")
data_inflasi = st.number_input('Enter Inflation Data', format="%.4f")

# Memungkinkan pengguna memilih model yang akan digunakan untuk prediksi
model_option = st.selectbox("Choose Model for Prediction:", ['Decision Tree', 'Random Forest', 'AdaBoost'])

# Menentukan model berdasarkan pilihan pengguna
if model_option == 'Decision Tree':
    model = dt_model
elif model_option == 'Random Forest':
    model = rf_model
else:  # 'AdaBoost'
    model = ada_model

# Tombol untuk melakukan prediksi
predict_btn = st.button("Predict Price")

# Prediksi dan menampilkan hasil
if predict_btn:
    # Membuat prediksi
    inputs = np.array([[ihsg, kurs_jual, data_inflasi]])
    predicted_price = model.predict(inputs)[0]
    
    # Menampilkan hasil prediksi
    st.write("")
    st.subheader(f"Predicted Gold Price: Rp {predicted_price:,.2f} using {model_option}")
