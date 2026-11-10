import os
import io
import base64
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Penting untuk backend Flask
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey_bps_siantar'
DB_PATH = 'users.db'

# ==================== DATABASE SETUP ====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

# ==================== HTML TEMPLATES ====================
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}Dashboard{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f7f6; }
        .navbar-brand { font-weight: bold; }
        .card { border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    {% if session.get('user_id') %}
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="#">Data Mining BPS Pematangsiantar</a>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><span class="nav-link text-light">Halo, {{ session.get('username') }}</span></li>
                    <li class="nav-item"><a class="nav-link" href="{{ url_for('logout') }}">Logout</a></li>
                </ul>
            </div>
        </div>
    </nav>
    {% endif %}
    
    <div class="container mt-4">
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for msg in messages %}<div class="alert alert-info">{{ msg }}</div>{% endfor %}
            {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

LOGIN_TEMPLATE = """
{% extends base %}
{% block title %}Login{% endblock %}
{% block content %}
<div class="row justify-content-center mt-5">
    <div class="col-md-4">
        <div class="card p-4">
            <h3 class="text-center mb-4">Login Sistem</h3>
            <form method="POST">
                <div class="mb-3"><label>Username</label><input type="text" name="username" class="form-control" required></div>
                <div class="mb-3"><label>Password</label><input type="password" name="password" class="form-control" required></div>
                <button type="submit" class="btn btn-primary w-100">Login</button>
            </form>
            <p class="text-center mt-3">Belum punya akun? <a href="{{ url_for('register') }}">Register di sini</a></p>
        </div>
    </div>
</div>
{% endblock %}
"""

REGISTER_TEMPLATE = """
{% extends base %}
{% block title %}Register{% endblock %}
{% block content %}
<div class="row justify-content-center mt-5">
    <div class="col-md-4">
        <div class="card p-4">
            <h3 class="text-center mb-4">Registrasi Analis</h3>
            <form method="POST">
                <div class="mb-3"><label>Username</label><input type="text" name="username" class="form-control" required></div>
                <div class="mb-3"><label>Password</label><input type="password" name="password" class="form-control" required></div>
                <button type="submit" class="btn btn-success w-100">Register</button>
            </form>
            <p class="text-center mt-3">Sudah punya akun? <a href="{{ url_for('login') }}">Login</a></p>
        </div>
    </div>
</div>
{% endblock %}
"""

DASHBOARD_TEMPLATE = """
{% extends base %}
{% block title %}Dashboard Analisis{% endblock %}
{% block content %}
<ul class="nav nav-tabs mb-4" id="mainTab" role="tablist">
    <li class="nav-item" role="presentation">
        <button class="nav-link active" id="kmeans-tab" data-bs-toggle="tab" data-bs-target="#kmeans" type="button">1. K-Means Clustering (PPT Workflow)</button>
    </li>
    <li class="nav-item" role="presentation">
        <button class="nav-link" id="prophet-tab" data-bs-toggle="tab" data-bs-target="#prophet" type="button">2. Proyeksi TPT (Prophet)</button>
    </li>
    <li class="nav-item" role="presentation">
        <button class="nav-link" id="paten-tab" data-bs-toggle="tab" data-bs-target="#paten" type="button">3. Tinjauan Paten Algoritmik</button>
    </li>
</ul>

<div class="tab-content" id="mainTabContent">
    <!-- TAB 1: K-MEANS (Sesuai Alur PPT) -->
    <div class="tab-pane fade show active" id="kmeans" role="tabpanel">
        <div class="card p-4 mb-4">
            <h5>Alur Kerja K-Means (Input CSV -> Parameter K -> Eksekusi -> Visualisasi)</h5>
            <form method="POST" enctype="multipart/form-data" class="row g-3 mt-2">
                <div class="col-md-6">
                    <label class="form-label">1. Upload Dataset (CSV)</label>
                    <input type="file" name="file" class="form-control" accept=".csv" required>
                </div>
                <div class="col-md-3">
                    <label class="form-label">2. Parameter K (Jumlah Klaster)</label>
                    <input type="number" name="k" class="form-control" min="1" max="10" value="3" required>
                </div>
                <div class="col-md-3 d-flex align-items-end">
                    <button type="submit" class="btn btn-primary w-100">3. Eksekusi K-Means</button>
                </div>
            </form>
        </div>

        {% if plot_url %}
        <div class="card p-4 mb-4">
            <h5>4. Visualisasi & Log Hasil</h5>
            <p class="text-muted">File berhasil dimuat! Baris: {{ rows }}, Kolom: {{ cols }}</p>
            <div class="text-center mb-3">
                <img src="data:image/png;base64,{{ plot_url }}" class="img-fluid" alt="Scatter Plot">
            </div>
            <h6>Tabel Hasil Klasterisasi:</h6>
            <div class="table-responsive" style="max-height: 300px; overflow-y: auto;">
                <table class="table table-striped table-sm">
                    <thead><tr>{% for col in table_headers %}<th>{{ col }}</th>{% endfor %}</tr></thead>
                    <tbody>
                        {% for row in table_data %}
                        <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}
    </div>

    <!-- TAB 2: PROPHET (Placeholder sesuai Judul) -->
    <div class="tab-pane fade" id="prophet" role="tabpanel">
        <div class="card p-4">
            <h5>Proyeksi Tingkat Pengangguran Terbuka (TPT) - Python Prophet</h5>
            <p>Modul ini akan memproses data deret waktu (Time Series) TPT Kota Pematangsiantar menggunakan algoritma <b>Facebook Prophet</b> untuk memproyeksikan tren pengangguran di masa depan.</p>
            <div class="alert alert-secondary">Status: <i>Menunggu integrasi dataset Time Series BPS (ds, y).</i></div>
        </div>
    </div>

    <!-- TAB 3: PATEN (Placeholder sesuai Judul) -->
    <div class="tab-pane fade" id="paten" role="tabpanel">
        <div class="card p-4">
            <h5>Tinjauan Paten Sederhana Pada Model Algoritmik</h5>
            <p>Dokumentasi dan tinjauan hak kekayaan intelektual (HKI) / Paten Sederhana atas modifikasi model algoritmik K-Means dan Prophet yang diterapkan pada kasus statistik daerah.</p>
            <div class="alert alert-secondary">Status: <i>Draft dokumen paten sedang disiapkan.</i></div>
        </div>
    </div>
</div>
{% endblock %}
"""

# ==================== ROUTES ====================
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            flash('Registrasi berhasil! Silakan login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username sudah digunakan.')
        finally:
            conn.close()
    return render_template_string(REGISTER_TEMPLATE, base=BASE_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah.')
    return render_template_string(LOGIN_TEMPLATE, base=BASE_TEMPLATE)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    plot_url = None
    table_headers = []
    table_data = []
    rows, cols = 0, 0

    if request.method == 'POST':
        file = request.files['file']
        k = int(request.form['k'])
        
        if file:
            # 1. Input CSV
            df = pd.read_csv(file)
            rows, cols = df.shape
            
            # 2. & 3. Eksekusi K-Means
            # Ambil hanya kolom numerik untuk clustering
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) >= 2 and k <= len(df):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(numeric_df)
                df['Klaster'] = labels
                
                # 4. Visualisasi (Scatter Plot Matplotlib)
                x_col = numeric_df.columns[0]
                y_col = numeric_df.columns[1]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(df[x_col], df[y_col], c=df['Klaster'], cmap='viridis', label='Data')
                ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                           s=200, c='red', marker='X', label='Centroids')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'K-Means Clustering (K={k})')
                ax.legend()
                
                # Save plot to base64
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close()
                
                # Prepare Table Data
                table_headers = df.columns.tolist()
                table_data = df.values.tolist()
            else:
                flash('CSV harus memiliki minimal 2 kolom numerik dan K tidak boleh lebih dari jumlah baris.')

    return render_template_string(DASHBOARD_TEMPLATE, base=BASE_TEMPLATE, 
                                  plot_url=plot_url, table_headers=table_headers, 
                                  table_data=table_data, rows=rows, cols=cols)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)