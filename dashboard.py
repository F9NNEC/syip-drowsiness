from flask import Flask, Response, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import timedelta
import os
import threading
from dotenv import load_dotenv

from microcontroller import init_esp32, get_gps_data
from inference import (
    gen_frames_stream,
    calibration_status,
    calibration_status_lock,
    request_calibration,
    start_detection,
    cleanup
)

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
            #map {
                width: 100%;
                height: 100%;
                background: #dfe8ea;
            }
            .location-meta {
                position: absolute;
                left: 12px;
                bottom: 12px;
                background: rgba(255,255,255,0.9);
                border-radius: 10px;
                padding: 8px 10px;
                font-size: 11px;
                color: #333;
                z-index: 500;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
        </style>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
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
                <div id="map"></div>
                <div class="location-meta" id="locationMeta">Menunggu GPS...</div>
            </div>

        </div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
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
            let completionTimer = null;

            function startRecalibrate() {
                clearTimeout(completionTimer);
                recalibrateBtn.disabled = true;
                recalibrateBtn.textContent = 'Mengkalibrasi...';
                recalibrateTrack.style.display = 'block';
                recalibrateStatusText.style.display = 'block';
                recalibrateStatusText.textContent = 'Mempersiapkan kalibrasi...';
                recalibrateFill.style.width = '0%';

                fetch('/recalibrate', { method: 'POST' })
                    .then(res => res.json())
                    .then(() => {
                        startCalibrationPolling();
                    })
                    .catch(() => {
                        recalibrateStatusText.textContent = 'Gagal memulai kalibrasi. Coba lagi.';
                        resetRecalibrateButton();
                    });
            }

            function startCalibrationPolling() {
                if (pollTimer === null) {
                    pollTimer = setInterval(pollCalibrationStatus, 400);
                }
                pollCalibrationStatus();
            }

            function pollCalibrationStatus() {
                fetch('/calibration_status')
                    .then(res => res.json())
                    .then(data => {
                        if (data.state === 'idle') return;

                        recalibrateBtn.disabled = true;
                        recalibrateBtn.textContent = 'Mengkalibrasi...';
                        recalibrateTrack.style.display = 'block';
                        recalibrateStatusText.style.display = 'block';
                        const pct = Math.round((data.progress / data.total) * 100);
                        recalibrateFill.style.width = pct + '%';
                        recalibrateStatusText.textContent =
                            `Mengumpulkan data wajah... ${data.progress}/${data.total}`;

                        if (data.state === 'done') {
                            clearInterval(pollTimer);
                            pollTimer = null;
                            recalibrateStatusText.textContent = 'Kalibrasi selesai!';
                            completionTimer = setTimeout(() => {
                                recalibrateTrack.style.display = 'none';
                                recalibrateStatusText.style.display = 'none';
                                resetRecalibrateButton();
                                completionTimer = null;
                            }, 1500);
                        }
                    })
                    .catch(() => {
                        if (pollTimer !== null) {
                            clearInterval(pollTimer);
                            pollTimer = null;
                            recalibrateStatusText.textContent = 'Gagal mengambil status kalibrasi.';
                            resetRecalibrateButton();
                        }
                    });
            }

            startCalibrationPolling();

            function resetRecalibrateButton() {
                recalibrateBtn.disabled = false;
                recalibrateBtn.textContent = 'Kalibrasi Ulang';
            }

            // --- GPS Map ---
            const map = L.map('map', { zoomControl: false }).setView([-2.885477, 104.715871], 14);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);

            let marker = L.marker([-2.885477, 104.715871]).addTo(map);

            function updateGpsMap(data) {
                if (!data || data.lat == null || data.lng == null) return;

                const latlng = [data.lat, data.lng];
                marker.setLatLng(latlng);
                map.setView(latlng, 16);

                const speedText = data.speed != null ? data.speed.toFixed(2) + ' km/h' : 'speed n/a';
                document.getElementById('locationMeta').textContent =
                    `Lat: ${data.lat.toFixed(5)}, Lng: ${data.lng.toFixed(5)} | ${speedText}`;
            }

            async function fetchGps() {
                try {
                    const res = await fetch('/gps_data');
                    const data = await res.json();
                    updateGpsMap(data);
                } catch (err) {
                    console.log('GPS belum tersedia');
                }
            }

            setInterval(fetchGps, 1000);
            fetchGps();
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
    if not request_calibration():
        return {'status': 'already_running'}, 200
    return {'status': 'started'}, 200


@app.route('/calibration_status')
@login_required
def calibration_status_route():
    """ Dipanggil berkala (polling) dari web untuk menampilkan progress. """
    with calibration_status_lock:
        return dict(calibration_status)


@app.route('/gps_data')
@login_required
def gps_data():
    return get_gps_data()


if __name__ == '__main__':
    start_detection()

    print('[INFO] Server Flask berjalan di http://127.0.0.1:5000')
    threading.Timer(1.5, lambda: init_esp32(port='COM3')).start()

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print('\n[INFO] Server dihentikan.')
    finally:
        cleanup()
        print('[INFO] Cleanup selesai.')
