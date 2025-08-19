import os
import pandas as pd
import joblib
from datetime import datetime
from flask_login import current_user

CSV_PATH = os.path.join("data", "history.csv")

def load_model(path):
    return joblib.load(path)

def predict_input(model, input_data):
    return model.predict([input_data])[0]

def save_to_csv(username, feature1, feature2, feature3, cluster):
    """
    Simpan riwayat prediksi ke CSV, menyimpan username (bukan user_id),
    dan menjaga maksimal 10 entri per user.
    """
    os.makedirs("data", exist_ok=True)

    new_row = {
        "username": username,
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "cluster": cluster,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["username", "feature1", "feature2", "feature3", "cluster", "timestamp"])

    # Tambahkan data baru di urutan teratas
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)

    # Batasi hanya 10 data terbaru per user
    df = (
        df.groupby("username", group_keys=False)
        .apply(lambda x: x.head(10))
        .reset_index(drop=True)
    )

    df.to_csv(CSV_PATH, index=False)

def load_user_history(username):
    """
    Ambil riwayat prediksi untuk user tertentu, maksimal 10, urut terbaru.
    """
    if not os.path.exists(CSV_PATH):
        return []

    df = pd.read_csv(CSV_PATH)
    df = df[df["username"] == username]  # filter hanya milik user
    df = df.head(10)  # maksimal 10
    return df.to_dict(orient="records")
