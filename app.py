# ========================
# Standard Library Imports
# ========================
import os
import io
import csv
import uuid
import base64
import locale
from functools import wraps
from datetime import datetime

# ========================
# Third-Party Library Imports
# ========================
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, jsonify, Response, abort, session
)
from flask_login import (
    LoginManager, login_user, login_required, logout_user, current_user
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# ========================
# Internal Project Imports
# ========================
from forms import LoginForm, RegisterForm, ClusteringForm
from models import db, User, Clustering, History, LoginHistory

# ========================
# Configurations
# ========================
matplotlib.use("Agg")
locale.setlocale(locale.LC_TIME, "indonesian")

# -------------------------
# Basic configuration
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "eureka123"
USERS_DB = os.path.join(BASE_DIR, "app.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{USERS_DB}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# Lokasi file CSV di folder data
HISTORY_FILE = os.path.join("data", "history.csv")
os.makedirs("data", exist_ok=True)

# Login Manager
login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------
# Auth routes
# -------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    form = RegisterForm()
    if form.validate_on_submit():
        nama_lengkap = form.nama_lengkap.data.strip()
        username = form.username.data.strip()
        password = form.password.data
        if User.query.filter_by(username=username).first():
            flash("Username sudah terpakai.", "error")
            return redirect(url_for("register"))
        hashed = generate_password_hash(password)
        user = User(
            nama_lengkap=nama_lengkap,
            username=username,
            password=hashed,
            role="pegawai"
        )
        db.session.add(user)
        db.session.commit()
        flash("Registrasi berhasil. Silakan login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    # Jika user sudah login, langsung ke halaman predict
    if current_user.is_authenticated:
        return redirect(url_for("predict"))

    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data

        # Ambil user dari database
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            # Login user
            login_user(user)

            # Simpan riwayat login
            ip_addr = request.remote_addr
            login_log = LoginHistory(
                user_id=user.id,
                username=user.username,
                ip_address=ip_addr
            )
            db.session.add(login_log)
            db.session.commit()

            # Set session agar bisa dipakai di template
            session['user_id'] = user.id
            session['role'] = user.role
            session['username'] = user.username
            session['nama_lengkap'] = user.nama_lengkap

            flash("Login berhasil!", "success")

            # Redirect ke halaman berikutnya
            next_page = request.args.get("next")
            return redirect(next_page or url_for("predict"))
        else:
            flash("Username atau password salah.", "error")

    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Anda telah logout.", "info")
    return redirect(url_for("login"))
# -------------------------
# Clustering routes
# -------------------------

@app.route("/clustering", methods=["GET", "POST"])
@login_required
def predict():
    form = ClusteringForm()
    clustering = None

    # Pastikan file CSV ada dengan header
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["feature1", "feature2", "feature3", "cluster", "timestamp"])

    if form.validate_on_submit():
        # Ambil input user
        user_input = {
            "Nilai Pajak": form.feature1.data,
            "Jenis Penerimaan Pajak": form.feature2.data,
            "Kota/Kab": form.feature3.data
        }

        # Prediksi cluster
        df_in = pd.DataFrame([user_input])
        X_scaled, _ = preprocess_for_model(df_in)
        preds = xgb_model.predict(X_scaled)
        labels = le.inverse_transform(preds)
        cluster_result = labels[0]

        clustering = f"Wajib Pajak termasuk dalam Cluster {cluster_result}"

        # Simpan ke CSV
        with open(HISTORY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                form.feature1.data,
                form.feature2.data,
                form.feature3.data,
                cluster_result,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    # Baca riwayat dari CSV
    history_df = pd.read_csv(HISTORY_FILE)
    history = history_df.to_dict(orient="records")

    return render_template("clustering.html", form=form, clustering=clustering, history=history)

@app.route('/somepage')
def somepage():
    current_time = datetime.now()
    return render_template('layout.html', now=current_time)

@app.route("/info-cluster")
@login_required  # atau hapus jika semua user bisa lihat tanpa login
def info_cluster():
    return render_template("info_cluster.html")

@app.context_processor
def inject_now():
    return {'now': datetime.now()} 

# -------------------------
# Izin khusus untuk admin
# -------------------------

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            flash("Anda tidak memiliki akses ke halaman ini.", "error")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/manage-users')
@login_required
@admin_required
def manage_users():
    users = User.query.all()  # asumsi model User ada
    return render_template('admin/manage_users.html', users=users)

# -------------------------
# Form Clustering
# -------------------------

@app.route("/form-clustering", methods=["GET", "POST"])
@login_required
def form_clustering():
    form = ClusteringForm()
    prediction = None

    # Ambil riwayat CSV milik user login, max 10
    csv_history = load_user_history(current_user.username)

    # Ambil riwayat DB milik user login, max 10
    db_history = (
        History.query
        .filter(History.user_id == current_user.id)
        .order_by(History.timestamp.desc())
        .limit(10)
        .all()
    )

    if form.validate_on_submit():
        nilai_pajak = form.feature1.data
        jenis_pajak = form.feature2.data
        kota_kab = form.feature3.data

        # Prediksi cluster
        user_input = {
            "Nilai Pajak": nilai_pajak,
            "Jenis Penerimaan Pajak": jenis_pajak,
            "Kota/Kab": kota_kab
        }
        df_in = pd.DataFrame([user_input])
        X_scaled, _ = preprocess_for_model(df_in)
        preds = xgb_model.predict(X_scaled)
        labels = le.inverse_transform(preds)
        cluster = int(labels[0])
        prediction = f"Wajib Pajak termasuk dalam Cluster {cluster}"

        # Simpan ke DB
        new_entry = History(
            user_id=current_user.id,
            feature1=nilai_pajak,
            feature2=jenis_pajak,
            feature3=kota_kab,
            cluster=cluster,
            timestamp=datetime.utcnow()
        )
        db.session.add(new_entry)
        db.session.commit()

        # Hapus data lama kalau lebih dari 10 (DB)
        total_entries = History.query.filter_by(user_id=current_user.id).count()
        if total_entries > 10:
            oldest_entries = (
                History.query
                .filter_by(user_id=current_user.id)
                .order_by(History.timestamp.asc())
                .limit(total_entries - 10)
                .all()
            )
            for entry in oldest_entries:
                db.session.delete(entry)
            db.session.commit()

        # Simpan ke CSV
        save_to_csv(
            username=current_user.username,  # pakai username
            feature1=nilai_pajak,
            feature2=jenis_pajak,
            feature3=kota_kab,
            cluster=cluster
        )

        return redirect(url_for("form_clustering"))

    # Kosongkan form saat reload
    form.feature1.data = ""
    form.feature2.data = ""
    form.feature3.data = ""

    return render_template(
        "clustering.html",
        form=form,
        prediction=prediction,
        history=db_history,  # atau csv_history kalau mau ambil dari CSV
        csv_history=csv_history
    )

# -------------------------
# Mengatur User
# -------------------------

@app.route('/update-role/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def update_role(user_id):
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    if new_role not in ['admin', 'pegawai']:
        flash('Role tidak valid.', 'error')
        return redirect(url_for('manage_users'))
    user.role = new_role
    db.session.commit()
    flash(f'Role user {user.username} berhasil diperbarui.', 'success')
    return redirect(url_for('manage_users'))

@app.route('/delete-user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.username} berhasil dihapus.', 'success')
    return redirect(url_for('manage_users'))


# -------------------------
# Prediksi PDRB Time Series
# -------------------------

# Load pickle model dan data
sarima_path = os.path.join(MODEL_DIR, "sarima_model.pkl")
ts_data_path = os.path.join(MODEL_DIR, "ts_data_processed.pkl")

for p in (sarima_path, ts_data_path):
    if not os.path.exists(p):
        raise RuntimeError(f"Required model artifact not found: {p}")

sarima_model = joblib.load(sarima_path)
ts_data_preprocess = joblib.load(ts_data_path)

# Pastikan ts_data berbentuk Series dengan index datetime
if isinstance(ts_data_preprocess, pd.DataFrame):
    if 'Date' in ts_data_preprocess.columns:
        ts_data_preprocess['Date'] = pd.to_datetime(ts_data_preprocess['Date'])
        ts_data_preprocess.set_index('Date', inplace=True)
    ts_data = ts_data_preprocess['Nilai_winsorized']
elif isinstance(ts_data_preprocess, pd.Series):
    ts_data = ts_data_preprocess
    if not isinstance(ts_data.index, pd.DatetimeIndex):
        ts_data.index = pd.to_datetime(ts_data.index)
else:
    raise ValueError("ts_data_preprocess harus berupa DataFrame atau Series")

@app.route("/sarima-predict")
@login_required
def sarima_predict():
    try:
        # ===== Bagian evaluasi (tetap untuk data test) =====
        n_forecast = len(ts_data)
        forecast_res = sarima_model.get_forecast(steps=n_forecast)
        forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        # Samakan index untuk evaluasi
        common_idx = ts_data.index.intersection(forecast.index)
        test = ts_data.loc[common_idx]
        forecast = forecast.loc[common_idx]

        if len(test) == 0 or len(forecast) == 0:
            flash("Data test atau hasil prediksi kosong, tidak dapat menghitung metrik evaluasi.", "error")
            return render_template("sarima_predict.html", mse=None, rmse=None, mae=None, plot_data=None)

        mse = mean_squared_error(test, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, forecast)

        mse = round(mse, 4)
        rmse = round(rmse, 4)
        mae = round(mae, 4)

        # ===== Perpanjang prediksi hingga 2028 =====
        forecast_end = pd.to_datetime("2028-12-31")
        steps_to_2028 = ((forecast_end.year - ts_data.index[-1].year) * 12) + \
                        (forecast_end.month - ts_data.index[-1].month)

        forecast_long_res = sarima_model.get_forecast(steps=steps_to_2028)
        forecast_long = forecast_long_res.predicted_mean
        conf_int_long = forecast_long_res.conf_int()

        # Set index prediksi panjang
        forecast_long.index = pd.date_range(
            start=ts_data.index[-1] + pd.offsets.MonthBegin(),
            periods=steps_to_2028, freq='MS'
        )
        conf_int_long.index = forecast_long.index

        # ===== Plot gabungan =====
        plt.figure(figsize=(12,6))
        plt.plot(ts_data.index, ts_data, label='Data Aktual')
        plt.plot(forecast_long.index, forecast_long, label='Forecast hingga 2028')
        plt.fill_between(conf_int_long.index, conf_int_long.iloc[:,0], conf_int_long.iloc[:,1],
                         color='pink', alpha=0.3, label='Confidence Interval')
        plt.ylim(-10, 20)  # batas y-axis
        plt.title('Prediksi Pertumbuhan Ekonomi')
        plt.xlabel('Date')
        plt.ylabel('Nilai Winsorized')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode()

        return render_template(
            "sarima_predict.html",
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            plot_data=plot_data
        )

    except Exception as e:
        flash(f"Terjadi error saat prediksi: {e}", "error")
        return render_template("sarima_predict.html", mse=None, rmse=None, mae=None, plot_data=None)

# -------------------------
# Load ML artifacts
# -------------------------
xgb_path = os.path.join(MODEL_DIR, "xgb_taxpredict_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
training_cols_path = os.path.join(MODEL_DIR, "training_columns.pkl")

for p in (xgb_path, scaler_path, le_path, training_cols_path):
    if not os.path.exists(p):
        raise RuntimeError(f"Required model artifact not found: {p}")

xgb_model = joblib.load(xgb_path)
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)
training_columns = joblib.load(training_cols_path)

REQUIRED_COLUMNS = ["Nilai Pajak", "Jenis Penerimaan Pajak", "Kota/Kab"]

# -------------------------
# Helper utilities
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/clustering-data/<filename>')
def get_clustering_data(filename):
    # Pastikan validasi filename supaya aman, jangan akses file sembarangan
    filepath = f'upload/result/{filename}'
    try:
        df = pd.read_csv(filepath)
    except Exception:
        abort(404, description="File tidak ditemukan atau gagal dibaca")

    # Contoh: kirim seluruh data sebagai list dict (hati-hati kalau datanya besar)
    data = df.to_dict(orient='records')
    columns = list(df.columns)

    return jsonify({
        "columns": columns,
        "data": data
    })

def preprocess_for_model(df):
    df_proc = df.copy()
    if "Nilai Pajak" in df_proc.columns:
        df_proc["Nilai Pajak"] = (
            df_proc["Nilai Pajak"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .replace("", "0")
        )
        df_proc["Nilai Pajak"] = pd.to_numeric(df_proc["Nilai Pajak"], errors="coerce").fillna(0)

    categorical_candidates = [c for c in ["Jenis Penerimaan Pajak", "Kota/Kab"] if c in df_proc.columns]
    df_cat = pd.get_dummies(df_proc[categorical_candidates].astype(str)) if categorical_candidates else pd.DataFrame(index=df_proc.index)

    num_cols = [c for c in ["Nilai Pajak"] if c in df_proc.columns]
    df_num = df_proc[num_cols].reset_index(drop=True) if num_cols else pd.DataFrame(index=df_proc.index)

    df_encoded = pd.concat([df_num, df_cat], axis=1).reset_index(drop=True)
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    X_scaled = scaler.transform(df_aligned)

    return X_scaled, df_aligned

def format_currency_series(series):
    def fmt(x):
        try:
            if pd.isna(x):
                return ""
            if float(x).is_integer():
                return f"{int(x):,}"
            return f"{x:,.2f}"
        except Exception:
            return str(x)
    return series.apply(lambda x: f"Rp {x:,.2f}" if pd.notnull(x) else "")

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("Tidak ada file terlampir", "error")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("Nama file kosong", "error")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uid = uuid.uuid4().hex[:8]
            saved_name = f"{os.path.splitext(filename)[0]}_{uid}.csv"
            saved_path = os.path.join(RESULTS_DIR, saved_name)

            try:
                df = pd.read_csv(file)
            except Exception as e:
                flash(f"Gagal membaca CSV: {e}", "error")
                return redirect(request.url)

            if not all(col in df.columns for col in REQUIRED_COLUMNS):
                missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                flash(f"Kolom wajib tidak ada: {', '.join(missing)}", "error")
                return redirect(request.url)

            X_scaled, _ = preprocess_for_model(df)
            preds = xgb_model.predict(X_scaled)
            labels = le.inverse_transform(preds)

            df_with_pred = df.copy()
            df_with_pred["Predicted_Cluster"] = labels
            df_with_pred.to_csv(saved_path, index=False)

            flash(f"File berhasil diproses. Hasil: {saved_name}", "success")
            return redirect(url_for("upload_result", filename=saved_name))
        else:
            flash("Format file tidak didukung. Gunakan CSV.", "error")
            return redirect(request.url)

    return render_template("upload.html")

@app.route("/upload/result/<filename>")
@login_required
def upload_result(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        flash("File tidak ditemukan", "error")
        return redirect(url_for("upload"))

    df = pd.read_csv(file_path, nrows=100)
    if "Nilai Pajak" in df.columns:
        df["Nilai Pajak"] = format_currency_series(df["Nilai Pajak"])

    preview_html = df.to_html(classes="min-w-full table-auto", index=False, escape=False)
    return render_template("upload_result.html", filename=filename, preview=preview_html)

@app.route("/delete-file/<filename>", methods=["POST"])
@login_required
def delete_file(filename):
    if session.get("role") != "admin":  # hanya admin
        return jsonify({"error": "Unauthorized"}), 403

    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path) and file_path.endswith(".csv"):
        os.remove(file_path)
        return jsonify({"success": True, "message": f"{filename} berhasil dihapus"})
    else:
        return jsonify({"error": "File tidak ditemukan"}), 404


@app.route("/download/<filename>")
@login_required
def download_file(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

@app.route("/stats")
@login_required
def stats_page():
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".csv")],
        key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)),
        reverse=True
    )
    latest = files[0] if files else None
    current_time = datetime.now()
    return render_template("stats.html", latest_filename=latest, files=files, now=current_time)

@app.route("/api/stats/<filename>")
@login_required
def api_stats(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "file not found"}), 404

    df = pd.read_csv(file_path)

    # exclude Predicted_Cluster from numeric stats
    numeric = df.select_dtypes(include=["number"]).drop(columns=["Predicted_Cluster"], errors="ignore")
    descr = numeric.describe().reset_index().to_dict(orient="records")

    numeric_values = {col: numeric[col].dropna().tolist() for col in numeric.columns}

    categorical = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        counts = df[col].value_counts(dropna=False).head(20)
        categorical[col] = [{"label": str(i), "count": int(v)} for i, v in counts.items()]

    corr = numeric.corr().fillna(0).to_dict()

    return jsonify({
        "describe": descr,
        "categorical": categorical,
        "corr": corr,
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_values": numeric_values
    })

@app.route("/api/stats/download/<filename>")
@login_required
def api_stats_download(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        flash("File tidak ditemukan", "error")
        return redirect(url_for("stats_page"))

    df = pd.read_csv(file_path)
    numeric = df.select_dtypes(include=["number"]).drop(columns=["Predicted_Cluster"], errors="ignore")
    descr = numeric.describe().reset_index()

    csv_buf = io.StringIO()
    descr.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=stats_{filename}"}
    )

@app.route("/api/cluster-viz/<filename>")
@login_required
def api_cluster_viz(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "file not found"}), 404

    df = pd.read_csv(file_path)

    required_cols = ["Nilai Pajak", "Jenis Penerimaan Pajak", "Kota/Kab", "Predicted_Cluster"]
    if not all(col in df.columns for col in required_cols):
        return jsonify({"error": "required columns missing"}), 400

    df_proc = df.copy()
    df_proc["Nilai Pajak"] = (
        df_proc["Nilai Pajak"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("", "0")
    )
    df_proc["Nilai Pajak"] = pd.to_numeric(df_proc["Nilai Pajak"], errors="coerce").fillna(0)

    cat_cols = ["Jenis Penerimaan Pajak", "Kota/Kab"]
    df_cat = pd.get_dummies(df_proc[cat_cols].astype(str))

    features = pd.concat([df_proc[["Nilai Pajak"]], df_cat], axis=1)

    X_scaled = scaler.transform(features.reindex(columns=training_columns, fill_value=0))

    # Ganti TSNE dengan PCA 2D
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X_scaled)

    clusters = df["Predicted_Cluster"].astype(str).tolist()

    tsne_points = [{"x": float(x), "y": float(y), "cluster": c} for x, y, c in zip(pca_results[:,0], pca_results[:,1], clusters)]

    cluster_counts = df.groupby(["Kota/Kab", "Predicted_Cluster"]).size().reset_index(name="count")
    cluster_counts_dict = {}
    for _, row in cluster_counts.iterrows():
        kota = row["Kota/Kab"]
        cluster = str(row["Predicted_Cluster"])
        count = int(row["count"])
        cluster_counts_dict.setdefault(kota, []).append({"cluster": cluster, "count": count})

    return jsonify({
        "tsne": tsne_points,  # tetap pakai nama key "tsne" agar frontend gak perlu diubah
        "cluster_counts": cluster_counts_dict
    })

@app.route("/dashboard")
@login_required
def dashboard():
    try:
        files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".csv")],
            key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)),
            reverse=True
        )
    except FileNotFoundError:
        files = []

    latest_file = files[0] if files else None

    data = None
    if latest_file:
        df = pd.read_csv(os.path.join(RESULTS_DIR, latest_file))
        if "Nilai Pajak" in df.columns:
            df["Nilai Pajak"] = format_currency_series(df["Nilai Pajak"])
        data = df.to_dict(orient="records")

    return render_template("dashboard.html", data=data)

@app.route("/export")
@login_required
@admin_required
def export():
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".csv")],
        key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)),
        reverse=True
    )
    if not files:
        flash("Tidak ada file untuk diunduh.", "error")
        return redirect(url_for("dashboard"))
    latest_file = files[0]
    return send_from_directory(RESULTS_DIR, latest_file, as_attachment=True)

@app.route("/admin/dashboard")
@login_required
@admin_required
def login_history():
    # Ambil query parameter untuk filter
    page = request.args.get("page", 1, type=int)
    start_date_str = request.args.get("start_date", "")
    end_date_str = request.args.get("end_date", "")

    query = LoginHistory.query

    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            query = query.filter(LoginHistory.login_time >= start_date)
        except ValueError:
            pass

    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            end_date = end_date.replace(hour=23, minute=59, second=59)
            query = query.filter(LoginHistory.login_time <= end_date)
        except ValueError:
            pass

    logs = query.order_by(LoginHistory.login_time.desc()).paginate(page=page, per_page=10)

    return render_template(
        "admin/login_history.html",
        logs=logs,
        start_date=start_date_str,
        end_date=end_date_str
    )

# -------------------------
# Decorator role-based access
# -------------------------
from flask import abort

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)  # Forbidden
        return f(*args, **kwargs)
    return decorated_function

# =====================================================
# PETA INTERAKTIF
# =====================================================

@app.route("/map")
@login_required
def map_page():
    # ambil file hasil clustering terbaru
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.lower().endswith(".csv")],
        key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)),
        reverse=True
    )
    latest_file = files[0] if files else None
    return render_template("map.html", latest_filename=latest_file)


@app.route("/api/map-data/<filename>")
@login_required
def api_map_data(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "file not found"}), 404
    
    df = pd.read_csv(file_path)
    if "Kota/Kab" not in df.columns or "Predicted_Cluster" not in df.columns:
        return jsonify({"error": "required columns missing"}), 400

    # Hitung jumlah per cluster untuk tiap kota/kab
    grouped = (
        df.groupby(["Kota/Kab", "Predicted_Cluster"])
          .size()
          .reset_index(name="count")
    )

    # Susun dict: { "Kota/Kab": {cluster: count, ...}, ... }
    result = {}
    for _, row in grouped.iterrows():
        kota = row["Kota/Kab"]
        cluster = int(row["Predicted_Cluster"])
        count = int(row["count"])
        if kota not in result:
            result[kota] = {}
        result[kota][cluster] = count

    # konversi ke list agar mudah di frontend
    data = [{"Kota/Kab": kota, "clusters": clusters} for kota, clusters in result.items()]
    return jsonify(data)


@app.route("/api/cluster-summary/<filename>")
@login_required
def api_cluster_summary(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "file not found"}), 404

    df = pd.read_csv(file_path)
    if "Predicted_Cluster" not in df.columns:
        return jsonify({"error": "Predicted_Cluster column not found"}), 400

    # hitung total per cluster
    counts = df["Predicted_Cluster"].value_counts().to_dict()
    return jsonify(counts)

# -------------------------
# Initialize DB & optional admin user
# -------------------------
def ensure_admin_user():
    if User.query.filter_by(role="admin").count() == 0:
        admin = User(
            nama_lengkap="Administrator",
            username="admin",
            password=generate_password_hash("admin123"),
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()
        app.logger.info("Default admin created: admin/admin123")

with app.app_context():
    db.create_all()
    ensure_admin_user()

print(app.url_map)
# -------------------------
# Run
# -------------------------

@app.route("/")
def home():
    return render_template("clustering.html")

if __name__ == "__main__":
    # Buat database jika belum ada
    if not os.path.exists(USERS_DB):
        with app.app_context():
            db.create_all()
    app.run(debug=True)