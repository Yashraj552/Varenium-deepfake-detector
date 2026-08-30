import hashlib
import os
import sys
import re
import secrets
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
import cv2
import numpy as np
from flask import Flask, request, jsonify, session, send_from_directory, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageOps, ImageFilter
import onnxruntime as ort
from werkzeug.security import check_password_hash, generate_password_hash
import logging
from dotenv import load_dotenv
from marshmallow import Schema, fields, ValidationError
import bcrypt
from authlib.integrations.flask_client import OAuth
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

load_dotenv()

def _env_bool(name, default=False):
    value = os.environ.get(name, str(default)).strip().lower()
    return value in {'1', 'true', 'yes', 'on'}


def _env_list(name, default):
    value = os.environ.get(name)
    if value:
        return [item.strip() for item in value.split(',') if item.strip()]
    return default


LOGIN_FAILURE_WINDOW_SECONDS = 900
LOGIN_FAILURE_LIMIT = 5
login_failures = defaultdict(list)

# Setup Secure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("secure_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
import io
import base64

app = Flask(__name__)
allowed_origins = _env_list('CORS_ALLOWED_ORIGINS', [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://localhost:5001',
    'http://127.0.0.1:5001',
    'https://deepfake-detector-rosy.vercel.app'
])
CORS(app,
     resources={r"/*": {"origins": allowed_origins}},
     supports_credentials=True,
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

_is_production = os.environ.get('FLASK_ENV') != 'development' and not os.environ.get('DEBUG')
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(64),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=os.environ.get('SESSION_COOKIE_SAMESITE', 'None' if _is_production else 'Lax'),
    SESSION_COOKIE_SECURE=_env_bool('SESSION_COOKIE_SECURE', True if _is_production else False),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
)

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=(), payment=(), usb=()'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin-allow-popups'
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    if request.is_secure:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

firebase_app = None
# Support JSON content via env var (Railway) or file path (local dev)
_firebase_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
_firebase_file = os.environ.get('FIREBASE_SERVICE_ACCOUNT_FILE')
try:
    if _firebase_json:
        import json as _json
        _cred_dict = _json.loads(_firebase_json)
        firebase_app = firebase_admin.initialize_app(credentials.Certificate(_cred_dict))
    elif _firebase_file and os.path.exists(_firebase_file):
        firebase_app = firebase_admin.initialize_app(credentials.Certificate(_firebase_file))
except Exception as _e:
    print(f"[!] Firebase init failed: {_e}")

oauth = OAuth(app)
if os.environ.get('GOOGLE_CLIENT_ID') and os.environ.get('GOOGLE_CLIENT_SECRET'):
    oauth.register(
        name='google',
        client_id=os.environ['GOOGLE_CLIENT_ID'],
        client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:8000')


def cleaup_login_attempts(email):
    now = datetime.utcnow().timestamp()
    attempts = login_failures.get(email, [])
    login_failures[email] = [ts for ts in attempts if now - ts <= LOGIN_FAILURE_WINDOW_SECONDS]


def is_login_rate_limited(email):
    cleaup_login_attempts(email)
    return len(login_failures.get(email, [])) >= LOGIN_FAILURE_LIMIT


def record_failed_login(email):
    now = datetime.utcnow().timestamp()
    attempts = login_failures.get(email, [])
    attempts.append(now)
    login_failures[email] = attempts[-LOGIN_FAILURE_LIMIT:]


def clear_login_failures(email):
    login_failures.pop(email, None)

# Global Exception Handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled Exception: {str(e)}", exc_info=True)
    return jsonify({"error": "An internal server error occurred. Please try again later."}), 500


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "Resource not found."}), 404

# Limit uploads to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
# Database configuration for Supabase Postgres
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'forensic.db'))
# Supabase uses connection pooling, sometimes requires specific connect args depending on setup, but SQLAlchemy usually handles it fine.
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Allowed file types
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

db = SQLAlchemy(app)

AUTH_DB_PATH = os.path.join(os.path.dirname(__file__), 'auth.db')


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    investigations = db.relationship('Investigation', backref='user', lazy=True)
    audit_logs = db.relationship('AuditLog', backref='user', lazy=True)


class Investigation(db.Model):
    __tablename__ = 'investigations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(128), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=True) # Stateless storage
    analysis_result = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(200), nullable=False)
    resource = db.Column(db.String(255), nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status = db.Column(db.String(50), nullable=False)
    details = db.Column(db.Text, nullable=True)


with app.app_context():
    db.create_all()


def ensure_investigation_schema():
    db_path = os.path.join(os.path.dirname(__file__), 'forensic.db')
    if not os.path.exists(db_path):
        return

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute('PRAGMA table_info(investigations)').fetchall()}
        if 'image_data' not in columns:
            connection.execute('ALTER TABLE investigations ADD COLUMN image_data BLOB')


with app.app_context():
    ensure_investigation_schema()


def get_auth_db():
    connection = sqlite3.connect(AUTH_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_db():
    with get_auth_db() as connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
                created_at TEXT NOT NULL
            )
        ''')
        columns = {column['name'] for column in connection.execute('PRAGMA table_info(users)')}
        if 'role' not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")


def migrate_legacy_users():
    if not os.path.exists(AUTH_DB_PATH):
        return
    with sqlite3.connect(AUTH_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        legacy_users = connection.execute(
            'SELECT id, name, email, password_hash, role, created_at FROM users'
        ).fetchall()

    for row in legacy_users:
        user = User.query.filter_by(email=row['email']).first()
        if user is None:
            db.session.add(User(
                name=row['name'],
                email=row['email'],
                password_hash=row['password_hash'],
                role=row['role'] or 'user',
                is_active=True,
                created_at=datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S') if isinstance(row['created_at'], str) else datetime.utcnow()
            ))
        else:
            user.role = row['role'] or user.role
            user.password_hash = row['password_hash']
    db.session.commit()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required.'}), 401
        return view(*args, **kwargs)
    return wrapped_view


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        user = User.query.get(session['user_id'])
        if user is None or user.role != 'admin':
            return jsonify({'error': 'Administrator access required.'}), 403
        return view(*args, **kwargs)
    return wrapped_view


initialize_auth_db()
with app.app_context():
    migrate_legacy_users()

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'best_deepfake_detector.onnx')

# --- LOAD MODEL ---
ort_session = None
if os.path.exists(MODEL_PATH):
    try:
        ort_session = ort.InferenceSession(MODEL_PATH)
        print(f"[*] ONNX Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"[!] Error loading ONNX model: {e}")
else:
    print(f"[!] Warning: ONNX Model weights not found at {MODEL_PATH}")

# Preprocessing Pipeline (NumPy Equivalent)
def preprocess_image(pil_img):
    img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_data = np.array(img).astype('float32') / 255.0
    img_data = np.transpose(img_data, (2, 0, 1)) # HWC to CHW
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img_data = (img_data - mean) / std
    return np.expand_dims(img_data, axis=0).astype(np.float32)

def get_image_metadata(img_path):
    img = Image.open(img_path)
    return {
        "format": img.format,
        "mode": img.mode,
        "size": f"{img.width}x{img.height}",
        "filesize_kb": round(os.path.getsize(img_path) / 1024, 2)
    }

def generate_frequency_map(input_tensor_np):
    # Simplified Haar/Frequency visualization approximation for ONNX pipeline
    # We take the input image, extract high frequencies via OpenCV Laplacian
    img = input_tensor_np[0].transpose((1, 2, 0)) # CHW to HWC
    # un-normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    freq_viz = np.abs(laplacian)
    freq_viz = (freq_viz - freq_viz.min()) / (freq_viz.max() - freq_viz.min() + 1e-8)
    freq_viz = (freq_viz * 255).astype(np.uint8)
    freq_viz = cv2.applyColorMap(freq_viz, cv2.COLORMAP_JET)
    freq_viz = cv2.resize(freq_viz, (224, 224))
    freq_viz = cv2.cvtColor(freq_viz, cv2.COLOR_BGR2RGB)
    return freq_viz

def generate_heatmap_onnx(face_img_pil, input_tensor_np, prob_fake):
    """
    ONNX-compatible heatmap using two real forensic techniques:
    1. Error Level Analysis (ELA) — reveals JPEG compression inconsistencies
       introduced by GAN/diffusion model synthesis.
    2. Local Variance Map — highlights unnaturally smooth or over-sharp regions
       typical of AI-generated faces.
    The two maps are fused and overlaid on the original face crop.
    """
    face_224 = face_img_pil.resize((224, 224))
    img_np = np.array(face_224).astype(np.float32)

    # --- Method 1: Error Level Analysis (ELA) ---
    ela_buffer = io.BytesIO()
    face_224.save(ela_buffer, format='JPEG', quality=75)
    ela_buffer.seek(0)
    recompressed = np.array(Image.open(ela_buffer)).astype(np.float32)
    # Amplify the difference so subtle artifacts are visible
    ela_diff = np.abs(img_np - recompressed)
    ela_map = np.mean(ela_diff, axis=2)  # collapse channels → single channel
    ela_map = ela_map * 12.0             # amplification factor
    ela_map = np.clip(ela_map, 0, 255)

    # --- Method 2: Local Variance Map ---
    # High local variance = sharp edges (natural); low variance = over-smooth (AI skin)
    # We invert it so AI-smooth regions light up on the heatmap
    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ksize = 9
    mean_local = cv2.blur(gray, (ksize, ksize))
    sq_mean_local = cv2.blur(gray ** 2, (ksize, ksize))
    local_var = np.clip(sq_mean_local - mean_local ** 2, 0, None)
    # Invert: smooth (low variance) = suspicious
    inv_var_map = 1.0 / (local_var + 1.0)
    inv_var_map = (inv_var_map - inv_var_map.min()) / (inv_var_map.max() - inv_var_map.min() + 1e-8)
    inv_var_map = inv_var_map * 255.0

    # --- Normalise each map to [0, 255] ---
    ela_norm = (ela_map - ela_map.min()) / (ela_map.max() - ela_map.min() + 1e-8) * 255.0

    # --- Fuse: weight ELA higher when model is confident of fake ---
    ela_weight = 0.55 + 0.2 * prob_fake  # 0.55 – 0.75
    var_weight = 1.0 - ela_weight
    combined = ela_weight * ela_norm + var_weight * inv_var_map

    # Gaussian smoothing for visual clarity
    combined = cv2.GaussianBlur(combined.astype(np.float32), (21, 21), 0)
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    # Apply JET colormap (red = high suspicion, blue = low)
    heatmap_colored = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original face image
    # More opaque overlay when model thinks it's fake
    alpha = min(0.35 + 0.35 * prob_fake, 0.65)
    blended = cv2.addWeighted(img_np.astype(np.uint8), 1.0 - alpha,
                              heatmap_colored, alpha, 0)
    return blended

def generate_reasoning(prob_fake, logit_mag, heatmap_img, freq_img):
    spatial_intensity = np.mean(heatmap_img) / 255.0
    freq_volatility = np.std(freq_img) / 255.0
    if prob_fake >= 0.5:
        headline = "The image shows signals that can be associated with digital manipulation."
        summary = "The model found enough unusual visual and texture patterns to flag this image for review."
        signals = [
            "Some regions differ from the surrounding image texture.",
            "The fine-detail pattern is not fully consistent across the image.",
        ]
        next_step = "Treat this as a review lead, not final proof. Check the original file and its source."
    else:
        headline = "The image did not show strong manipulation signals in this scan."
        summary = "The model found the image broadly consistent with the examples it learned as authentic."
        signals = [
            "The visible texture pattern appears broadly consistent across the image.",
            "No strong synthetic signal crossed the model's decision threshold.",
        ]
        next_step = "Do not treat this as proof that the image is genuine, especially if it is edited, stylized, or AI-assisted."

    if logit_mag < 2.0:
        certainty = "The result is borderline, so extra review is especially important."
    elif logit_mag < 8.0:
        certainty = "The model's signal is moderate rather than decisive."
    else:
        certainty = "The model's signal is comparatively strong, but it can still be wrong outside its training data."

    report = [headline, summary, *signals, certainty, next_step]
    return {
        "headline": headline,
        "summary": summary,
        "signals": signals,
        "certainty": certainty,
        "next_step": next_step,
        "text": "\n\n".join(report),
        "map_context": {
            "spatial_intensity": round(float(spatial_intensity), 3),
            "frequency_variation": round(float(freq_volatility), 3),
        },
    }

def run_forensic_analysis(img_path):
    original_img = Image.open(img_path).convert('RGB')
    metadata = get_image_metadata(img_path)
    
    # Match the full-image resize and normalization used during validation.
    face_img = original_img

    input_tensor_np = preprocess_image(face_img)
    
    if ort_session is not None:
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor_np}
        out = ort_session.run(None, ort_inputs)[0]
    else:
        out = np.array([[0.0, 0.0]])
        
    # Softmax
    exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
    probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
    
    prob_fake = float(probs[0][1])
    logit_mag = float(np.abs(out[0][0] - out[0][1]))

    # Keep the production decision threshold aligned with model validation.
    threshold = 0.5
    is_fake = prob_fake > threshold
    
    # Generate Visualizations
    # ELA + Local Variance forensic heatmap (ONNX-compatible, no backprop needed)
    heatmap_img = generate_heatmap_onnx(face_img, input_tensor_np, prob_fake)
    freq_img = generate_frequency_map(input_tensor_np)
    
    # Generate DYNAMIC XAI Reasoning based on Heatmap and Frequency statistics
    reasoning = generate_reasoning(prob_fake, logit_mag, heatmap_img, freq_img)
    
    def to_b64(img_np):
        pil_img = Image.fromarray(img_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "prediction": "MANIPULATED" if is_fake else "AUTHENTIC",
        "confidence": round(prob_fake * 100, 2) if is_fake else round((1 - prob_fake) * 100, 2),
        "heatmap": to_b64(heatmap_img),
        "frequency_map": to_b64(freq_img),
        "reasoning": reasoning["text"],
        "reasoning_detail": reasoning,
        "metadata": metadata,
        "mode": "Deep Scan" if logit_mag < 8.0 else "Standard",
        "logit_mag": round(logit_mag, 2)
    }

@app.route('/')
def home():
    return jsonify({"status": "FORENSIC_KERNEL_ACTIVE", "version": "2.1.0"})


class RegisterSchema(Schema):
    name = fields.Str(required=True, validate=lambda x: 2 <= len(x) <= 100)
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=lambda x: 8 <= len(x) <= 128)
    confirm_password = fields.Str(required=True)

class LoginSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True)

@app.route('/auth/register', methods=['POST'])
def register():
    payload = request.get_json(silent=True) or {}
    try:
        data = RegisterSchema().load(payload)
    except ValidationError as err:
        return jsonify({'error': 'Invalid input', 'details': err.messages}), 400

    if data['password'] != data['confirm_password']:
        return jsonify({'error': 'Passwords do not match.'}), 400

    email = data['email'].lower()
    existing_user = User.query.filter_by(email=email).first()
    if existing_user is not None:
        logger.warning(f"Registration attempt for existing email: {email}")
        return jsonify({'error': 'An account with that email already exists.'}), 409

    hashed = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    user = User(
        name=data['name'],
        email=email,
        password_hash=hashed,
        role='user',
        is_active=True,
        created_at=datetime.utcnow(),
    )
    db.session.add(user)
    
    # Audit Log
    audit = AuditLog(user_id=None, action="USER_REGISTER", status="SUCCESS", details=f"User {email} registered", ip_address=request.remote_addr)
    db.session.add(audit)
    db.session.commit()
    logger.info(f"New user registered: {email}")
    return jsonify({'message': 'Registration successful.'}), 201


@app.route('/auth/login', methods=['POST'])
def login():
    payload = request.get_json(silent=True) or {}
    try:
        data = LoginSchema().load(payload)
    except ValidationError as err:
        return jsonify({'error': 'Invalid input', 'details': err.messages}), 400

    email = data['email'].lower()

    if is_login_rate_limited(email):
        logger.warning(f"Rate-limited login attempt for email: {email}")
        return jsonify({'error': 'Too many failed login attempts. Please try again in a few minutes.'}), 429

    user = User.query.filter_by(email=email).first()
    
    if user is None:
        record_failed_login(email)
        logger.warning(f"Failed login attempt for non-existent email: {email}")
        return jsonify({'error': 'Invalid email or password.'}), 401

    # Check bcrypt or legacy pbkdf2
    is_valid = False
    if user.password_hash.startswith('pbkdf2:sha256'):
        is_valid = check_password_hash(user.password_hash, data['password'])
    else:
        is_valid = bcrypt.checkpw(data['password'].encode('utf-8'), user.password_hash.encode('utf-8'))

    if not is_valid:
        record_failed_login(email)
        logger.warning(f"Failed login attempt for email: {email}")
        audit = AuditLog(user_id=user.id, action="USER_LOGIN", status="FAILED", details="Invalid password", ip_address=request.remote_addr)
        db.session.add(audit)
        db.session.commit()
        return jsonify({'error': 'Invalid email or password.'}), 401

    clear_login_failures(email)
    user.last_login = datetime.utcnow()
    db.session.commit()

    session.clear()
    session.permanent = True
    session.modified = True
    session['user_id'] = user.id
    session['user_name'] = user.name
    session['user_email'] = user.email
    session['user_role'] = user.role
    db.session.add(AuditLog(
        user_id=user.id,
        action="USER_LOGIN",
        status="SUCCESS",
        details="Password authentication succeeded",
        ip_address=request.remote_addr,
    ))
    db.session.commit()
    return jsonify({'message': 'Login successful.', 'user': {'name': user.name, 'email': user.email, 'role': user.role}})


@app.route('/auth/firebase', methods=['POST'])
def firebase_login():
    if firebase_app is None:
        return jsonify({'error': 'Firebase sign-in is not configured on the server.'}), 503

    payload = request.get_json(silent=True) or {}
    id_token = payload.get('id_token')
    if not isinstance(id_token, str) or not id_token.strip():
        return jsonify({'error': 'Firebase identity token is required.'}), 400

    try:
        decoded = firebase_auth.verify_id_token(id_token, app=firebase_app)
        email = (decoded.get('email') or '').lower()
        if not email or decoded.get('email_verified') is not True:
            return jsonify({'error': 'A verified Firebase email is required.'}), 401

        user = User.query.filter_by(email=email).first()
        if user is None:
            random_password = bcrypt.hashpw(secrets.token_bytes(32), bcrypt.gensalt()).decode('utf-8')
            user = User(
                name=(decoded.get('name') or email.split('@')[0])[:100],
                email=email,
                password_hash=random_password,
                role='user',
                is_active=True,
                created_at=datetime.utcnow(),
            )
            db.session.add(user)

        if not user.is_active:
            return jsonify({'error': 'This account is inactive.'}), 403

        user.last_login = datetime.utcnow()
        db.session.commit()
        session.clear()
        session.permanent = True
        session['user_id'] = user.id
        session['user_name'] = user.name
        session['user_email'] = user.email
        session['user_role'] = user.role
        return jsonify({'message': 'Firebase login successful.', 'user': {'name': user.name, 'email': user.email, 'role': user.role}})
    except Exception:
        logger.exception('Firebase sign-in verification failed')
        return jsonify({'error': 'Firebase sign-in could not be verified.'}), 401


@app.route('/auth/google')
def google_login():
    google = oauth.create_client('google')
    if google is None:
        return jsonify({'error': 'Google sign-in is not configured on the server.'}), 503
    redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI') or url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    google = oauth.create_client('google')
    if google is None:
        return redirect(f'{FRONTEND_URL}/login?error=Google+sign-in+is+not+configured')

    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info or not user_info.get('email') or not user_info.get('email_verified'):
            raise ValueError('Google did not return a verified email address.')

        email = user_info['email'].lower()
        user = User.query.filter_by(email=email).first()
        if user is None:
            random_password = bcrypt.hashpw(secrets.token_bytes(32), bcrypt.gensalt()).decode('utf-8')
            user = User(
                name=(user_info.get('name') or email.split('@')[0])[:100],
                email=email,
                password_hash=random_password,
                role='user',
                is_active=True,
                created_at=datetime.utcnow(),
            )
            db.session.add(user)

        if not user.is_active:
            return redirect(f'{FRONTEND_URL}/login?error=This+account+is+inactive')

        user.last_login = datetime.utcnow()
        db.session.commit()
        session.clear()
        session.permanent = True
        session['user_id'] = user.id
        session['user_name'] = user.name
        session['user_email'] = user.email
        session['user_role'] = user.role
        return redirect(f'{FRONTEND_URL}/dashboard')
    except Exception:
        logger.exception('Google sign-in callback failed')
        return redirect(f'{FRONTEND_URL}/login?error=Google+sign-in+could+not+be+completed')


@app.route('/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logout successful.'})


@app.route('/auth/me')
def current_user():
    if 'user_id' not in session:
        return jsonify({'authenticated': False}), 401
    return jsonify({
        'authenticated': True,
        'user': {'name': session['user_name'], 'email': session['user_email'], 'role': session['user_role']}
    })


@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.order_by(User.id).all()
    return jsonify({'users': [{
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'role': user.role,
        'created_at': user.created_at.isoformat() if user.created_at else None
    } for user in users]})


@app.route('/investigations', methods=['GET'])
@login_required
def list_investigations():
    records = Investigation.query.filter_by(user_id=session['user_id']).order_by(Investigation.created_at.desc()).all()
    return jsonify({'investigations': [{
        'id': record.id,
        'original_filename': record.original_filename,
        'stored_filename': record.stored_filename,
        'file_hash': record.file_hash,
        'analysis_result': record.analysis_result,
        'confidence_score': record.confidence_score,
        'created_at': record.created_at.isoformat() if record.created_at else None
    } for record in records]})


@app.route('/investigations', methods=['DELETE'])
@login_required
def delete_investigations():
    records = Investigation.query.filter_by(user_id=session['user_id']).all()
    for record in records:
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'uploads', record.stored_filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        db.session.delete(record)
    db.session.commit()
    return jsonify({'message': 'Investigation history cleared.'})


@app.route('/uploads/<filename>')
@login_required
def serve_upload(filename):
    safe_name = os.path.basename(filename)
    record = Investigation.query.filter_by(stored_filename=safe_name).first()
    if record is None or record.user_id != session['user_id']:
        return jsonify({'error': 'Not authorized to access this file.'}), 403
    
    # If image_data is stored in DB (stateless mode)
    if record.image_data:
        from flask import send_file
        import mimetypes
        mime_type, _ = mimetypes.guess_type(safe_name)
        return send_file(io.BytesIO(record.image_data), mimetype=mime_type or 'image/jpeg')
        
    # Fallback for older records stored on disk
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    return send_from_directory(upload_dir, safe_name)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Validate presence
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    # Basic filename sanitization and extension check
    original_name = (file.filename or 'image.jpg').strip()[:255]
    _, ext = os.path.splitext(original_name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type."}), 400

    # Read into memory (MAX_CONTENT_LENGTH guards the total size)
    try:
        data = file.read()
    except Exception:
        return jsonify({"error": "Failed to read uploaded file."}), 400

    if not data:
        return jsonify({"error": "Uploaded file is empty."}), 400

    if len(data) > app.config.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024):
        return jsonify({"error": "Uploaded file exceeds allowed size."}), 413

    # Verify image using PIL to avoid trusting Content-Type
    try:
        img_buf = io.BytesIO(data)
        with Image.open(img_buf) as pil_img:
            pil_img.verify()
    except Exception:
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    # Run analysis using a temporary file
    import tempfile
    try:
        stored_name = f"{secrets.token_hex(16)}{ext}"
        file_hash = hashlib.sha256(data).hexdigest()
        
        # Write to temporary file for PyTorch CV2 processing
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(temp_fd, 'wb') as tmp:
            tmp.write(data)

        # Run analysis
        result = run_forensic_analysis(temp_path)
        
        # Cleanup temp file
        os.remove(temp_path)

        investigation = Investigation(
            user_id=session['user_id'],
            original_filename=original_name,
            stored_filename=stored_name,
            file_hash=file_hash,
            image_data=data, # Save directly to Postgres
            analysis_result=result['prediction'],
            confidence_score=float(result['confidence']),
            created_at=datetime.utcnow(),
        )
        db.session.add(investigation)
        db.session.commit()

        db.session.add(AuditLog(
            user_id=session['user_id'],
            action="IMAGE_ANALYSIS",
            status="SUCCESS",
            details=f"Analysis completed for {original_name}",
            ip_address=request.remote_addr,
        ))
        db.session.commit()

        result['investigation_id'] = investigation.id
        result['file_hash'] = file_hash
        return jsonify(result)

    except Exception as e:
        # Keep client-facing message generic; real details go to server logs
        logger.exception("Error during analysis/upload")
        # If temp_path exists, attempt cleanup
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return jsonify({"error": "Analysis failed. Please try again later."}), 500

if __name__ == '__main__':
    print("[*] Starting Forensic Lab Backend...")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
