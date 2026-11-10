import os
import io
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash

# ==================== KONFIGURASI ====================
st.set_page_config(page_title="Dashboard K-Means BPS", page_icon="📊", layout="wide")
DB_PATH = 'users.db'

# ==================== DATABASE ====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                     (username, generate_password_hash(password)))
        conn.commit()
        return True, "Registrasi berhasil!"
    except sqlite3.IntegrityError:
        return False, "Username sudah digunakan."
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    if user and check_password_hash(user[2], password):
        return True, user[1]
    return False, "Username atau password salah."

# ==================== HALAMAN LOGIN ====================
def login_page():
    st.title("🔐 Login Sistem Analisis Data")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            if username and password:
                success, msg = verify_user(username, password)
                if success:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = msg
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Isi username dan password.")
    
    st.markdown("---")
    if st.button("Belum punya akun? Klik di sini untuk Register"):
        st.session_state['page'] = 'register'
        st.rerun()

# ==================== HALAMAN REGISTER ====================
def register_page():
    st.title("📝 Registrasi Analis Baru")
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Konfirmasi Password", type="password")
        submitted = st.form_submit_button("Register", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("Password tidak cocok!")
            elif len(password) < 4:
                st.error("Password minimal 4 karakter.")
            else:
                success, msg = register_user(username, password)
                if success:
                    st.success(msg)
                    st.session_state['page'] = 'login'
                    st.rerun()
                else:
                    st.error(msg)
    
    if st.button("Kembali ke Login"):
        st.session_state['page'] = 'login'
        st.rerun()

# ==================== DASHBOARD ====================
def dashboard_page():
    st.sidebar.success(f"👤 Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
    
    st.title("📊 Dashboard Analisis Data Mining - BPS Pematangsiantar")
    
    tab1, tab2, tab3 = st.tabs([
        "1️⃣ K-Means Clustering (PPT Workflow)",
        "2️⃣ Proyeksi TPT (Prophet)",
        "3️⃣ Tinjauan Paten Algoritmik"
    ])
    
    # ========== TAB 1: K-MEANS ==========
    with tab1:
        st.header("Alur Kerja K-Means (Sesuai PPT)")
        st.markdown("**Input CSV → Parameter K → Eksekusi → Visualisasi**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("1️⃣ Upload Dataset (CSV)", type=['csv'])
        with col2:
            k_value = st.number_input("2️⃣ Jumlah Klaster (K)", min_value=1, max_value=10, value=3)
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil dimuat! Baris: {df.shape[0]}, Kolom: {df.shape[1]}")
            
            if st.button("3️⃣ Eksekusi K-Means", type="primary", use_container_width=True):
                numeric_df = df.select_dtypes(include=[np.number])
                
                if len(numeric_df.columns) < 2:
                    st.error("❌ CSV harus memiliki minimal 2 kolom numerik!")
                elif k_value > len(df):
                    st.error(f"❌ K ({k_value}) tidak boleh lebih dari jumlah baris ({len(df)})!")
                else:
                    # Eksekusi K-Means
                    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(numeric_df)
                    df['Klaster'] = labels
                    
                    st.markdown("---")
                    st.subheader("4️⃣ Visualisasi & Hasil")
                    
                    # Scatter Plot
                    x_col = numeric_df.columns[0]
                    y_col = numeric_df.columns[1]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(df[x_col], df[y_col], c=df['Klaster'], 
                                        cmap='viridis', s=100, alpha=0.7, edgecolor='k')
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                              s=300, c='red', marker='X', label='Centroids', edgecolor='k')
                    
                    # Label titik (jika kolom pertama adalah nama)
                    first_col = df.columns[0]
                    if df[first_col].dtype == 'object':
                        for i, txt in enumerate(df[first_col]):
                            ax.annotate(txt, (df[x_col].iloc[i], df[y_col].iloc[i]), 
                                       fontsize=8, alpha=0.7)
                    
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel(y_col, fontsize=12)
                    ax.set_title(f'K-Means Clustering (K={k_value})', fontsize=14, fontweight='bold')
                    ax.legend()
                    plt.colorbar(scatter, label='Klaster')
                    st.pyplot(fig)
                    
                    # Log Teks (sesuai format PPT)
                    st.markdown(f"### 📋 --- HASIL K-MEANS CLUSTERING (K={k_value}) ---")
                    log_text = ""
                    for idx, row in df.iterrows():
                        first_col_name = df.columns[0]
                        line = f"[{idx}] {row[first_col_name]} | "
                        for col in numeric_df.columns:
                            line += f"{col}: {row[col]} | "
                        line += f"Klaster: {row['Klaster']}"
                        log_text += line + "\n"
                    st.code(log_text, language="text")
                    
                    # Tabel Data
                    st.markdown("### 📊 Tabel Hasil Klasterisasi")
                    st.dataframe(df, use_container_width=True)
                    
                    # Statistik per klaster
                    st.markdown("### 📈 Ringkasan per Klaster")
                    summary = df.groupby('Klaster')[numeric_df.columns].mean()
                    st.dataframe(summary, use_container_width=True)
    
    # ========== TAB 2: PROPHET ==========
    with tab2:
        st.header("Proyeksi Tingkat Pengangguran Terbuka (TPT)")
        st.info("🔮 Modul ini akan menggunakan **Facebook Prophet** untuk memproyeksikan tren TPT Kota Pematangsiantar di masa depan berdasarkan data deret waktu historis.")
        st.warning("⏳ Status: Menunggu integrasi dataset Time Series BPS (format: kolom `ds` untuk tanggal, `y` untuk nilai TPT).")
    
    # ========== TAB 3: PATEN ==========
    with tab3:
        st.header("Tinjauan Paten Sederhana Pada Model Algoritmik")
        st.info("📜 Dokumentasi HKI / Paten Sederhana atas modifikasi model algoritmik K-Means dan Prophet yang diterapkan pada kasus statistik daerah.")
        st.warning("⏳ Status: Draft dokumen paten sedang disiapkan.")

# ==================== MAIN ROUTER ====================
def main():
    init_db()
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    
    if st.session_state['logged_in']:
        dashboard_page()
    elif st.session_state['page'] == 'register':
        register_page()
    else:
        login_page()

if __name__ == "__main__":
    main()