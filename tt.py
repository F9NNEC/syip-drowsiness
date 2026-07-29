import cv2
import mediapipe as mp
import math
import numpy as np
import time
import torch
import platform
from flask import Flask, Response
import serial

app = Flask(__name__)

backend = cv2.CAP_V4L2

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
    """ Proses kalibrasi secara diam-diam di background sebelum server start """
    ears = []
    mars = []
    pucs = []
    moes = []

    print('\n[INFO] Menjalankan Kalibrasi... Pastikan wajah ada di depan kamera.')
    
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    
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
    
    # PERBAIKAN: Ubah menjadi >= 4 (Mayoritas dari 6 chunk)
    return int(preds.sum() >= 4)

def gen_frames(ears_norm, mars_norm, pucs_norm, moes_norm):
    """ Generator fungsi untuk mengirim stream MJPEG ke Flask """
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
    
    epsilon = 1e-5 
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        ear, mar, puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ear = (ear - ears_norm[0]) / (ears_norm[1] + epsilon)
            mar = (mar - mars_norm[0]) / (mars_norm[1] + epsilon)
            puc = (puc - pucs_norm[0]) / (pucs_norm[1] + epsilon)
            moe = (moe - moes_norm[0]) / (moes_norm[1] + epsilon)
            
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

            # arduino
            if arduino is not None:
                if label == 1: 
                    arduino.write(b'1') # ngantuk
                else:          
                    arduino.write(b'0')


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
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- INISIALISASI VARIABEL GLOBAL ---
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
states = ['alert', 'drowsy']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

model_lstm_path = r'models/clf_lstm_jit6.pth'
model = torch.jit.load(model_lstm_path)
model.eval()

# --- INISIALISASI KONEKSI ARDUINO VIA USB ---
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    print("\n[INFO] Berhasil terhubung ke Arduino via USB.")
except Exception as e:
    arduino = None
    print(f"\n[WARNING] Arduino tidak terdeteksi. Error: {e}")

# Global norm variable untuk menampung hasil kalibrasi
ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g = None, None, None, None


# --- FLASK ROUTES ---
@app.route('/')
def index():
    return """
    <html>
      <body style="margin:0; background:#111; color:white; font-family:Arial; text-align:center;">
        <img id="video" src="/video_feed" style="width:100%; max-width:800px; border:2px solid #444;">
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Eksekusi kalibrasi terlebih dahulu sebelum server web menyala
    ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g = calibrate()
    
    print('[INFO] Memulai server Flask. Buka http://127.0.0.1:5000 di browser.')
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        face_mesh.close()