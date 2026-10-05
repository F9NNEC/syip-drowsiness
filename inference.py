import cv2
import mediapipe as mp
import math
import numpy as np
import time
import torch
import platform
import threading
from dotenv import load_dotenv
from microcontroller import get_esp32, close_esp32

load_dotenv()

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

def draw_head_pose(image, landmarks):
    height, width = image.shape[:2]
    image_points = np.array([
        [landmarks[index].x * width, landmarks[index].y * height]
        for index in [1, 152, 33, 263, 61, 291]
    ], dtype=np.float64)
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0]
    ], dtype=np.float64)

    focal_length = width
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch, yaw, roll = angles

    axis_length = 120
    axis = np.float64([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ])
    projected_axis, _ = cv2.projectPoints(
        axis, rotation_vector, translation_vector, camera_matrix, distortion
    )
    origin, x_axis, y_axis, z_axis = projected_axis.reshape(-1, 2).astype(int)
    origin = tuple(origin)
    axes = [
        (x_axis, (0, 0, 255), 'X'),
        (y_axis, (255, 0, 0), 'Z'),
        (z_axis, (0, 255, 0), 'Y')
    ]
    for endpoint, color, label in axes:
        endpoint = tuple(endpoint)
        cv2.arrowedLine(image, origin, endpoint, color, 2, tipLength=0.15)
        cv2.putText(image, label, endpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(
        image, f'Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}',
        (int(0.02 * width), int(0.94 * height)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )
    return pitch, yaw, roll

def run_face_mp(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    head_pose = None
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

            head_pose = draw_head_pose(image, face_landmarks.landmark)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar / ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image, head_pose

def collect_calibration_frames(cap, calib_frame_count=75):
    ears = []
    mars = []
    pucs = []
    moes = []

    print('\n[INFO] Menjalankan Kalibrasi... Pastikan wajah ada di depan kamera.')

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        ear, mar, puc, moe, _, _ = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)

            print(f'Progres Kalibrasi: {len(ears)}/{calib_frame_count} frame', end='\r')

        if len(ears) >= calib_frame_count:
            break

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
EYE_CLOSURE_THRESHOLD = -1.0
HEAD_DOWN_PITCH_MIN = -165.0
HEAD_DOWN_PITCH_MAX = -1.0
HEAD_UP_PITCH_MIN = 1.0
HEAD_UP_PITCH_MAX = 165.0
HEAD_TURN_YAW_THRESHOLD = 20.0

def get_detection_status(model_label, ear_value, head_pose):
    eye_closed = ear_value != -1000 and ear_value <= EYE_CLOSURE_THRESHOLD
    if head_pose is None:
        return 'normal'

    pitch, yaw, _ = head_pose
    looking_down = HEAD_DOWN_PITCH_MIN <= pitch <= HEAD_DOWN_PITCH_MAX
    looking_up = HEAD_UP_PITCH_MIN <= pitch <= HEAD_UP_PITCH_MAX
    looking_sideways = abs(yaw) > HEAD_TURN_YAW_THRESHOLD
    if looking_up or looking_sideways:
        return 'normal'

    if model_label == 1 and looking_down:
        return 'danger'
    if model_label == 1 or eye_closed:
        return 'drowsy'
    return 'normal'

norm_lock = threading.Lock()
current_norms = {'ears': None, 'mars': None, 'pucs': None, 'moes': None}

calibration_requested = threading.Event()
calibration_status_lock = threading.Lock()
calibration_status = {'state': 'idle', 'progress': 0, 'total': CALIB_FRAME_COUNT}
calib_buffer = {'ears': [], 'mars': [], 'pucs': [], 'moes': []}

def capture_loop(cap, ears_norm, mars_norm, pucs_norm, moes_norm):
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

    model_label = 0
    status = 'normal'
    last_sent_status = None
    input_data = []
    frame_before_run = 0

    epsilon = 1e-5

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # KALIBRASI ULANG
        if calibration_requested.is_set():
            ear_raw, mar_raw, puc_raw, moe_raw, calib_image, _ = run_face_mp(image)

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
                model_label = 0
                status = 'normal'

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

        ear, mar, puc, moe, image, head_pose = run_face_mp(image)
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
            model_label = get_classification(input_data)

        status = get_detection_status(model_label, ear_main, head_pose)
        if status != last_sent_status:
            esp = get_esp32()
            if esp and esp.is_connected():
                if status == 'danger':
                    sent = esp.send_danger()
                elif status == 'drowsy':
                    sent = esp.send_drowsy()
                else:
                    sent = esp.send_alert()
                if sent:
                    last_sent_status = status

        # Gambar teks indikator
        cv2.putText(image, "EAR: %.2f" % (ear_main), (int(0.02 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, "MAR: %.2f" % (mar_main), (int(0.27 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, "PUC: %.2f" % (puc_main), (int(0.52 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, "MOE: %.2f" % (moe_main), (int(0.77 * image.shape[1]), int(0.07 * image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if status != 'normal':
            color = (0, 0, 255) if status == 'danger' else (0, 165, 255)
            cv2.putText(image, status.upper(), (int(0.02 * image.shape[1]), int(0.2 * image.shape[0])),
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

# Global norm variable untuk menampung hasil kalibrasi
ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g = None, None, None, None
camera_cap = None

def start_detection():
    """Buka kamera sekali, lalu langsung kalibrasi dan masuk loop deteksi."""
    global ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g, camera_cap

    if camera_cap is not None and camera_cap.isOpened():
        camera_cap.release()

    camera_cap = cv2.VideoCapture(0, backend)
    camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g = collect_calibration_frames(camera_cap)

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(camera_cap, ears_norm_g, mars_norm_g, pucs_norm_g, moes_norm_g),
        daemon=True
    )
    capture_thread.start()


def cleanup():
    """Cleanup resources (close ESP32 connection, etc)."""
    global camera_cap
    if camera_cap is not None and camera_cap.isOpened():
        camera_cap.release()
    close_esp32()

if __name__ == '__main__':
    start_detection()
    print('[INFO] Deteksi wajah berjalan di background thread.')
    esp = get_esp32()
    status = "✓ Terhubung" if esp and esp.is_connected() else "✗ Tidak terhubung"
    print(f'[INFO] ESP32 {status}')