import cv2
import mediapipe as mp
import math
import numpy as np
import time
import torch
import platform
import threading
import os
from functools import wraps
from datetime import timedelta
from flask import Flask, Response, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import serial

load_dotenv()

app = Flask(__name__)

# KONFIGURASI LOGIN DASHBOARD
app.secret_key = os.environ.get('DASHBOARD_SECRET_KEY')
app.permanent_session_lifetime = timedelta(days=7)

DASHBOARD_USERNAME = os.environ.get('DASHBOARD_USERNAME')
DASHBOARD_PASSWORD_HASH = generate_password_hash(
    os.environ.get('DASHBOARD_PASSWORD')
)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

if platform.system() == 'Windows':
    backend = cv2.CAP_DSHOW  # Windows
else:
    backend = cv2.CAP_V4L2   # Raspberry Pi

def distance(p1, p2):
    return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5

def eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    return (eye_aspect_ratio(landmarks, left_eye) + \
            eye_aspect_ratio(landmarks, right_eye)) / 2

def mouth_feature(landmarks):
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def pupil_circularity(landmarks, eye):
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
                distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
                distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
                distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
                distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
                distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
                distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
                distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)

def pupil_feature(landmarks):
    return (pupil_circularity(landmarks, left_eye) + \
            pupil_circularity(landmarks, right_eye)) / 2

def run_face_mp(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z])
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar / ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image

def calibrate(calib_frame_count=75):
    ears = []
    mars = []
    pucs = []
    moes = []

    print('\n[INFO] Menjalankan Kalibrasi... Pastikan wajah ada di depan kamera.')
    
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        ear, mar, puc, moe, _ = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)
            
            print(f'Progres Kalibrasi: {len(ears)}/{calib_frame_count} frame', end='\r')

        if len(ears) >= calib_frame_count:
            break

    cap.release()
    print('\n[INFO] Kalibrasi Selesai!\n')
    
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)
    
    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
           [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]

def get_classification(input_data):
    model_input = []
    model_input.append(input_data[:5])
    model_input.append(input_data[3:8])
    model_input.append(input_data[6:11])
    model_input.append(input_data[9:14])
    model_input.append(input_data[12:17])
    model_input.append(input_data[15:])
    
    model_input = torch.FloatTensor(np.array(model_input))
    preds = torch.sigmoid(model(model_input)).gt(0.5).int().data.numpy()
    
    return int(preds.sum() >= 4)

# latest_frame menyimpan frame JPEG terakhir yang sudah diproses,
# supaya semua device (banyak viewer) tinggal "membaca" frame yang sama
# tanpa masing-masing membuka kamera / memanggil face_mesh sendiri-sendiri.
latest_frame = None
frame_lock = threading.Lock()

CALIB_FRAME_COUNT = 75

norm_lock = threading.Lock()
current_norms = {'ears': None, 'mars': None, 'pucs': None, 'moes': None}

calibration_requested = threading.Event()
calibration_status_lock = threading.Lock()
calibration_status = {'state': 'idle', 'progress': 0, 'total': CALIB_FRAME_COUNT}
calib_buffer = {'ears': [], 'mars': [], 'pucs': [], 'moes': []}

def capture_loop(ears_norm, mars_norm, pucs_norm, moes_norm):
    global latest_frame

    with norm_lock:
        current_norms['ears'] = ears_norm
        current_norms['mars'] = mars_norm
        current_norms['pucs'] = pucs_norm
        current_norms['moes'] = moes_norm

    ear_main = 0
    mar_main = 0
    puc_main = 0
    moe_main = 0
    decay = 0.9 

    label = None
    input_data = []
    frame_before_run = 0

    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    epsilon = 1e-5 
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # KALIBRASI ULANG
        if calibration_requested.is_set():
            ear_raw, mar_raw, puc_raw, moe_raw, calib_image = run_face_mp(image)

            if ear_raw != -1000:
                calib_buffer['ears'].append(ear_raw)
                calib_buffer['mars'].append(mar_raw)
                calib_buffer['pucs'].append(puc_raw)
                calib_buffer['moes'].append(moe_raw)

            progress = len(calib_buffer['ears'])
            with calibration_status_lock:
                calibration_status['progress'] = progress

            cv2.putText(calib_image, f"KALIBRASI ULANG... {progress}/{CALIB_FRAME_COUNT}",
                        (int(0.05 * calib_image.shape[1]), int(0.12 * calib_image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            if progress >= CALIB_FRAME_COUNT:
                ears_arr = np.array(calib_buffer['ears'])
                mars_arr = np.array(calib_buffer['mars'])
                pucs_arr = np.array(calib_buffer['pucs'])
                moes_arr = np.array(calib_buffer['moes'])

                with norm_lock:
                    current_norms['ears'] = [ears_arr.mean(), ears_arr.std()]
                    current_norms['mars'] = [mars_arr.mean(), mars_arr.std()]
                    current_norms['pucs'] = [pucs_arr.mean(), pucs_arr.std()]
                    current_norms['moes'] = [moes_arr.mean(), moes_arr.std()]

                calib_buffer['ears'].clear()
                calib_buffer['mars'].clear()
                calib_buffer['pucs'].clear()
                calib_buffer['moes'].clear()

                calibration_requested.clear()
                with calibration_status_lock:
                    calibration_status['state'] = 'done'
                    calibration_status['progress'] = CALIB_FRAME_COUNT

                ear_main = mar_main = puc_main = moe_main = 0
                input_data = []
                label = None

            ret, buffer = cv2.imencode('.jpg', calib_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with frame_lock:
                    latest_frame = buffer.tobytes()
            continue  # skip proses deteksi normal di bawah selama kalibrasi

        with norm_lock:
            ears_norm_local = current_norms['ears']
            mars_norm_local = current_norms['mars']
            pucs_norm_local = current_norms['pucs']
            moes_norm_local = current_norms['moes']

        ear, mar, puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ear = (ear - ears_norm_local[0]) / (ears_norm_local[1] + epsilon)
            mar = (mar - mars_norm_local[0]) / (mars_norm_local[1] + epsilon)
            puc = (puc - pucs_norm_local[0]) / (pucs_norm_local[1] + epsilon)
            moe = (moe - moes_norm_local[0]) / (moes_norm_local[1] + epsilon)
            
            if ear_main == -1000:
                ear_main = ear
                mar_main = mar
                puc_main = puc
                moe_main = moe
            else:
                ear_main = ear_main * decay + (1 - decay) * ear
                mar_main = mar_main * decay + (1 - decay) * mar
                puc_main = puc_main * decay + (1 - decay) * puc
                moe_main = moe_main * decay + (1 - decay) * moe
        else:
            ear_main = -1000
            mar_main = -1000
            puc_main = -1000
            moe_main = -1000

        if len(input_data) == 20:
            input_data.pop(0)
        input_data.append([ear_main, mar_main, puc_main, moe_main])

        frame_before_run += 1
        if frame_before_run >= 15 and len(input_data) == 20:
            frame_before_run = 0
            label = get_classification(input_data)

        frame_before_run += 1
        if frame_before_run >= 15 and len(input_data) == 20:
            frame_before_run = 0
            label = get_classification(input_data)

            # KIRIM SINYAL KE ARDUINO 
            if arduino is not None:
                if label == 1: 
                    arduino.write(b'1') # Kirim sinyal ngantuk
                else:          
                    arduino.write(b'0') # Kirim sinyal alert (bangun)

        # Gambar teks indikator
        cv2.putText(image, "EAR: %.2f" % (ear_main), (int(0.02 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "MAR: %.2f" % (mar_main), (int(0.27 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "PUC: %.2f" % (puc_main), (int(0.52 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, "MOE: %.2f" % (moe_main), (int(0.77 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        if label is not None:
            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.putText(image, "%s" % (states[label]), (int(0.02 * image.shape[1]), int(0.2 * image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

        # Encode gambar ke format JPEG
        ret, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue

        # Simpan sebagai frame terbaru (dibaca oleh semua viewer/device)
        with frame_lock:
            latest_frame = buffer.tobytes()

    cap.release()


def gen_frames_stream():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # batasi ~30fps per viewer, cegah CPU spinning


# INISIALISASI VARIABEL GLOBAL
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
states = ['alert', 'drowsy']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ganti path
model_lstm_path = r'models\clf_lstm_jit6.pth'
model = torch.jit.load(model_lstm_path)
model.eval()

# INISIALISASI KONEKSI ARDUINO VIA USB
try:
    # '/dev/ttyACM0' '/dev/ttyUSB0'
    arduino = serial.Serial('COM8', 9600, timeout=1)
    print("\n[INFO] Berhasil terhubung ke Arduino via USB.")
except Exception as e:
    arduino = None
    print(f"\n[WARNING] Arduino tidak terdeteksi. Error: {e}")

# Global norm variable untuk menampung hasil kalibrasi
ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g = None, None, None, None


# FLASK
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        if username == DASHBOARD_USERNAME and check_password_hash(DASHBOARD_PASSWORD_HASH, password):
            session.permanent = True
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Username atau password salah.'

    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Login - Monitoring Pengendara</title>
        <style>
            html, body {
                margin: 0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #f2f2f5;
                font-family: system-ui, -apple-system, sans-serif;
            }
            .login-card {
                background: #ffffff;
                width: 100%;
                max-width: 340px;
                padding: 32px 28px;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.06);
                box-sizing: border-box;
            }
            .login-card h1 {
                font-size: 22px;
                margin: 0 0 6px 0;
                color: #111;
            }
            .login-card p {
                font-size: 13px;
                color: #888;
                margin: 0 0 24px 0;
            }
            .field {
                margin-bottom: 16px;
            }
            .field label {
                display: block;
                font-size: 13px;
                font-weight: 600;
                color: #333;
                margin-bottom: 6px;
            }
            .field input {
                width: 100%;
                padding: 12px 14px;
                border-radius: 12px;
                border: 1px solid #e5e5e5;
                font-size: 14px;
                box-sizing: border-box;
            }
            .field input:focus {
                outline: none;
                border-color: #111;
            }
            .submit-btn {
                width: 100%;
                padding: 13px;
                border-radius: 12px;
                border: none;
                background-color: #111;
                color: #fff;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 8px;
            }
            .submit-btn:hover {
                background-color: #333;
            }
            .error-msg {
                background-color: #fdecec;
                color: #c0392b;
                font-size: 13px;
                padding: 10px 12px;
                border-radius: 10px;
                margin-bottom: 16px;
            }
        </style>
    </head>
    <body>
        <div class="login-card">
            <h1>Monitoring Pengendara</h1>
            <p>Masuk untuk melihat dashboard</p>
            """ + (f'<div class="error-msg">{error}</div>' if error else '') + """
            <form method="POST">
                <div class="field">
                    <label>Username</label>
                    <input type="text" name="username" autocomplete="username" required autofocus>
                </div>
                <div class="field">
                    <label>Password</label>
                    <input type="password" name="password" autocomplete="current-password" required>
                </div>
                <button class="submit-btn" type="submit">Masuk</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Monitoring Pengendara</title>
        <style>
            /* Reset dan styling dasar */
            html, body {
                margin: 0;
                background-color: #ffffff;
                font-family: system-ui, -apple-system, sans-serif;
                min-height: 100vh;
                overflow-x: hidden;
            }

            /* Wrapper untuk menampung container yang akan di-scale via JS */
            .scale-wrapper {
                width: 100%;
                display: flex;
                justify-content: center;
            }

            /* Kontainer utama: ukuran desain SELALU 390px (baku).
               Di desktop, tampil apa adanya (scale 1) sebagai kartu terpusat.
               Di HP, di-scale proporsional oleh JS supaya tampilan identik
               di semua ukuran layar -- bukan reflow ulang. */
            .mobile-container {
                background-color: #ffffff;
                width: 390px;
                box-sizing: border-box;
                padding: 24px;
                transform-origin: top center;
            }

            /* Header */
            .header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 30px;
            }
            .header-title {
                font-size: 26px;
                font-weight: 700;
                color: #000;
                line-height: 1.2;
                margin: 0;
            }
            .profile-btn {
                background-color: #f8f9fa;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-shrink: 0;
            }

            /* Dashcam Section */
            .dashcam-section {
                position: relative;
                margin-bottom: 24px;
                margin-top: 20px;
            }
            .badge-center {
                position: absolute;
                top: -14px;
                left: 50%;
                transform: translateX(-50%);
                background: white;
                padding: 4px 16px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                color: #333;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                z-index: 10;
            }
            .video-container {
                width: 100%;
                aspect-ratio: 16 / 9; /* sesuai resolusi asli kamera: 640x360 */
                border-radius: 20px;
                background-color: #f0f0f0;
                overflow: hidden;
            }
            .video-container img {
                width: 100%;
                height: 100%;
                object-fit: cover; /* aman, rasio kotak == rasio kamera, jadi tidak memotong */
            }

            /* Tombol & progress kalibrasi ulang */
            .recalibrate-wrap {
                margin-top: 10px;
            }
            .recalibrate-btn {
                width: 100%;
                padding: 10px;
                border-radius: 12px;
                border: 1px solid #e5e5e5;
                background-color: #fff;
                color: #111;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
            }
            .recalibrate-btn:hover {
                background-color: #f8f9fa;
            }
            .recalibrate-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .recalibrate-progress-track {
                width: 100%;
                height: 6px;
                border-radius: 6px;
                background-color: #eee;
                margin-top: 8px;
                overflow: hidden;
                display: none;
            }
            .recalibrate-progress-fill {
                height: 100%;
                width: 0%;
                background-color: #111;
                border-radius: 6px;
                transition: width 0.2s ease;
            }
            .recalibrate-status-text {
                font-size: 11px;
                color: #888;
                margin-top: 6px;
                text-align: center;
                display: none;
            }

            /* Heart Rate Section */
            .heart-card {
                border: 1px solid #f0f0f0;
                border-radius: 20px;
                padding: 20px;
                margin-bottom: 24px;
                display: flex;
                align-items: center;
                position: relative;
                box-shadow: 0 4px 12px rgba(0,0,0,0.02);
            }
            .heart-info {
                display: flex;
                flex-direction: column;
                min-width: 80px;
            }
            .heart-label {
                font-size: 14px;
                font-weight: 600;
                color: #000;
                margin-bottom: 5px;
            }
            .heart-value-container {
                display: flex;
                flex-direction: column;
            }
            .heart-value {
                font-size: 42px;
                font-weight: 700;
                color: #111;
                line-height: 1;
            }
            .heart-unit {
                font-size: 12px;
                color: #777;
                margin-top: 4px;
            }
            .heart-graph {
                flex-grow: 1;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 0 10px;
            }
            .view-graph-btn {
                position: absolute;
                top: 15px;
                right: 15px;
                border: 1px solid #eee;
                border-radius: 10px;
                padding: 6px 8px;
                display: flex;
                flex-direction: column;
                align-items: center;
                background: white;
            }
            .view-graph-btn span {
                font-size: 9px;
                font-weight: 600;
                color: #333;
                margin-top: 2px;
            }

            /* Location Section */
            .location-section {
                position: relative;
                width: 100%;
                height: 200px;
                border-radius: 20px;
                overflow: hidden;
                background-color: #eee;
            }
            .badge-left {
                position: absolute;
                top: 15px;
                left: 15px;
                background: white;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                color: #333;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                z-index: 10;
            }
            .location-section img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <div class="scale-wrapper">
        <div class="mobile-container">
            
            <div class="header">
                <h1 class="header-title">Monitoring<br>Pengendara</h1>
                <a href="/logout" class="profile-btn" title="Logout" style="text-decoration:none;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                        <circle cx="12" cy="7" r="4"></circle>
                    </svg>
                </a>
            </div>

            <div class="dashcam-section">
                <div class="badge-center">Dashcam</div>
                <div class="video-container">
                    <img id="video" src="/video_feed" alt="Video Feed">
                </div>

                <div class="recalibrate-wrap">
                    <button class="recalibrate-btn" id="recalibrateBtn" onclick="startRecalibrate()">
                        Kalibrasi Ulang
                    </button>
                    <div class="recalibrate-progress-track" id="recalibrateTrack">
                        <div class="recalibrate-progress-fill" id="recalibrateFill"></div>
                    </div>
                    <div class="recalibrate-status-text" id="recalibrateStatusText"></div>
                </div>
            </div>

            <div class="heart-card">
                <div class="heart-info">
                    <div class="heart-label">Heart</div>
                    <div class="heart-value-container">
                        <span class="heart-value">72</span>
                        <span class="heart-unit">bpm</span>
                    </div>
                </div>
                
                <div class="heart-graph">
                    <svg width="120" height="50" viewBox="0 0 120 50" stroke="#d94b4b" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="0,25 30,25 40,10 50,45 60,15 70,30 80,25 120,25" />
                    </svg>
                </div>

                <div class="view-graph-btn">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="#333">
                        <path d="M5 9h4v11H5zm6-6h4v17h-4zm6 10h4v7h-4z"/>
                    </svg>
                    <span>View Graph</span>
                </div>
            </div>

            <div class="location-section">
                <div class="badge-left">Location</div>
            </div>

        </div>
        </div>

        <script>
            // Desain dibuat pada lebar dasar 390px.
            // Di layar <= 430px (HP), scale menyesuaikan lebar device asli,
            // sehingga tampilan proporsinya SELALU identik di semua HP
            // (bukan reflow ulang yang membuat sebagian terlihat lebih besar/kecil).
            // Di layar > 430px (desktop), tetap tampil sebagai kartu 390px terpusat (scale 1).
            const BASE_WIDTH = 390;
            const MOBILE_BREAKPOINT = 430;

            function fitToScreen() {
                const container = document.querySelector('.mobile-container');
                const wrapper = document.querySelector('.scale-wrapper');
                const vw = window.innerWidth;

                const scale = vw <= MOBILE_BREAKPOINT ? (vw / BASE_WIDTH) : 1;
                container.style.transform = `scale(${scale})`;
                wrapper.style.height = (container.offsetHeight * scale) + 'px';
            }

            window.addEventListener('DOMContentLoaded', fitToScreen);
            window.addEventListener('resize', fitToScreen);
            fitToScreen(); // jalankan langsung juga, jangan tunggu 'load' (macet karena video_feed adalah stream MJPEG yang tak pernah "selesai")

            // --- Kalibrasi Ulang ---
            const recalibrateBtn = document.getElementById('recalibrateBtn');
            const recalibrateTrack = document.getElementById('recalibrateTrack');
            const recalibrateFill = document.getElementById('recalibrateFill');
            const recalibrateStatusText = document.getElementById('recalibrateStatusText');
            let pollTimer = null;

            function startRecalibrate() {
                recalibrateBtn.disabled = true;
                recalibrateBtn.textContent = 'Mengkalibrasi...';
                recalibrateTrack.style.display = 'block';
                recalibrateStatusText.style.display = 'block';
                recalibrateStatusText.textContent = 'Mempersiapkan kalibrasi...';
                recalibrateFill.style.width = '0%';

                fetch('/recalibrate', { method: 'POST' })
                    .then(res => res.json())
                    .then(() => {
                        pollTimer = setInterval(pollCalibrationStatus, 400);
                    })
                    .catch(() => {
                        recalibrateStatusText.textContent = 'Gagal memulai kalibrasi. Coba lagi.';
                        resetRecalibrateButton();
                    });
            }

            function pollCalibrationStatus() {
                fetch('/calibration_status')
                    .then(res => res.json())
                    .then(data => {
                        const pct = Math.round((data.progress / data.total) * 100);
                        recalibrateFill.style.width = pct + '%';
                        recalibrateStatusText.textContent =
                            `Mengumpulkan data wajah... ${data.progress}/${data.total}`;

                        if (data.state === 'done') {
                            clearInterval(pollTimer);
                            recalibrateStatusText.textContent = 'Kalibrasi selesai!';
                            setTimeout(() => {
                                recalibrateTrack.style.display = 'none';
                                recalibrateStatusText.style.display = 'none';
                                resetRecalibrateButton();
                            }, 1500);
                        }
                    })
                    .catch(() => {
                        clearInterval(pollTimer);
                        recalibrateStatusText.textContent = 'Gagal mengambil status kalibrasi.';
                        resetRecalibrateButton();
                    });
            }

            function resetRecalibrateButton() {
                recalibrateBtn.disabled = false;
                recalibrateBtn.textContent = 'Kalibrasi Ulang';
            }
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recalibrate', methods=['POST'])
@login_required
def recalibrate():
    if calibration_requested.is_set():
        return {'status': 'already_running'}, 200

    calib_buffer['ears'].clear()
    calib_buffer['mars'].clear()
    calib_buffer['pucs'].clear()
    calib_buffer['moes'].clear()

    with calibration_status_lock:
        calibration_status['state'] = 'calibrating'
        calibration_status['progress'] = 0
        calibration_status['total'] = CALIB_FRAME_COUNT

    calibration_requested.set()
    return {'status': 'started'}, 200

@app.route('/calibration_status')
@login_required
def calibration_status_route():
    """ Dipanggil berkala (polling) dari web untuk menampilkan progress. """
    with calibration_status_lock:
        return dict(calibration_status)

if __name__ == '__main__':
    ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g = calibrate()

    # Jalankan kamera + deteksi SEKALI di background thread.
    # Semua device yang buka dashboard (banyak viewer sekaligus) hanya
    # membaca latest_frame lewat gen_frames_stream(), bukan membuka kamera baru.
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g),
        daemon=True
    )
    capture_thread.start()

    print('[INFO] Memulai server Flask. Buka http://127.0.0.1:5000 di browser.')
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        face_mesh.close()