import sys
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FloatField
from wtforms.validators import DataRequired, Length, EqualTo

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

db = SQLAlchemy()

# ======================
# FORM REGISTER
# ======================
class RegisterForm(FlaskForm):
    nama_lengkap = StringField(
        'Nama Lengkap',
        validators=[DataRequired(), Length(min=3, max=100)]
    )
    username = StringField(
        'Username',
        validators=[DataRequired(), Length(min=3, max=50)]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired(), Length(min=8)]
    )
    confirm_password = PasswordField(
        'Konfirmasi Password',
        validators=[DataRequired(), EqualTo('password', message="Password tidak cocok.")]
    )
    submit = SubmitField('Register')


# ======================
# MODEL USER
# ======================
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    nama_lengkap = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="pegawai")

    # Relasi ke riwayat prediksi
    histories = db.relationship("History", backref="user", lazy=True, cascade="all, delete-orphan")

    @property
    def is_admin(self):
        return self.role.lower() == "admin"


# ======================
# MODEL HISTORY
# ======================
class History(db.Model):
    __tablename__ = "histories"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    feature1 = db.Column(db.String(100))
    feature2 = db.Column(db.String(100))
    feature3 = db.Column(db.String(100))
    cluster = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def timestamp_str(self):
        """Format tanggal agar langsung bisa dipakai di template."""
        if self.timestamp:
            return self.timestamp.strftime('%d %B %Y, %H:%M:%S')
        return "-"


# ======================
# MODEL PREDICTION (JIKA DIPAKAI TERPISAH)
# ======================
class Clustering(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    input1 = db.Column(db.Float, nullable=True)
    input2 = db.Column(db.Float, nullable=True)
    input3 = db.Column(db.Float, nullable=True)
    result = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("predictions", lazy=True, cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            "User ID": self.user_id,
            "Nama Lengkap": self.user.nama_lengkap if self.user else None,
            "Username": self.user.username if self.user else None,
            "Input 1": self.input1,
            "Input 2": self.input2,
            "Input 3": self.input3,
            "Result": self.result,
            "Created At": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

# ======================
# MODEL LOGIN HISTORY
# ======================
class LoginHistory(db.Model):
    __tablename__ = "login_histories"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    username = db.Column(db.String(50), nullable=False)
    ip_address = db.Column(db.String(45))
    login_time = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def login_time_str(self):
        if self.login_time:
            return self.login_time.strftime('%d %B %Y, %H:%M:%S')
        return "-"