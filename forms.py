from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField, PasswordField, SelectField
from wtforms.validators import DataRequired, Length, Regexp, EqualTo

class LoginForm(FlaskForm):
    """Form untuk login pengguna."""
    username = StringField(
        'Username',
        validators=[DataRequired(), Length(min=3, max=50)]
    )
    password = PasswordField(
        'Password',
        validators=[DataRequired()]
    )
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    """Form untuk registrasi pengguna baru (Menggunakan validasi baru yang lebih ketat)."""
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
        validators=[DataRequired(), Length(min=8), 
                    Regexp(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*\W)', 
                           message="Password harus mengandung huruf besar, huruf kecil, angka, dan simbol.")]
    )
    confirm_password = PasswordField(
        'Konfirmasi Password',
        validators=[DataRequired(), EqualTo('password', message='Password tidak cocok.')]
    )
    submit = SubmitField('Register')

    # FUNGSI PENTING DARI VERSI LAMA: untuk cek username duplikat di database
    def validate_username(self, username):
        """Memeriksa apakah username sudah ada di database."""
        user = User.query.filter_by(username=username.data.strip()).first()
        if user:
            raise ValidationError('Username ini sudah terpakai. Silakan pilih yang lain.')

class ClusteringForm(FlaskForm):
    feature1 = StringField(
        "Nilai Pajak",
        validators=[
            DataRequired(),
            Regexp(r"^[0-9,]+$", message="Hanya angka dan koma yang diperbolehkan.")
        ],
        render_kw={
            "placeholder": "Misal: 900,200,350"
        }
    )
    feature2 = StringField("Jenis Penerimaan Pajak", validators=[DataRequired()])
    feature3 = StringField("Kota/Kab", validators=[DataRequired()])
    submit = SubmitField("Submit")