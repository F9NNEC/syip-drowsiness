import cv2
import numpy as np
import time



# Inisialisasi kamera
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Set resolusi langsung dari hardware kamera (4:3)
for cap in [cap0, cap1]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def gen_frames():
    print("Start streaming...")
    target_fps = 20  # Batasi 20 FPS agar CPU Pi adem
    frame_time = 1.0 / target_fps

    while True:
        start_time = time.time()

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 and not ret1:
            time.sleep(0.01)
            continue

        # Jika kamera gagal dibaca, buat frame hitam 640x480
        if not ret0:
            frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            frame0 = cv2.flip(frame0, 1)

        if not ret1:
            frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            frame1 = cv2.flip(frame1, 1)

        # Gabungkan secara horizontal (Total resolusi: 1280x480)
        combined_frame = cv2.hconcat([frame0, frame1])

        # Kualitas JPEG diturunkan ke 35 untuk meringankan kompresi CPU
        ret, jpeg_buffer = cv2.imencode('.jpg', combined_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_buffer.tobytes() + b'\r\n')

        # Jeda sejenak untuk memberi nafas pada CPU
        elapsed = time.time() - start_time
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

def release_cameras():
    cap0.release()
    cap1.release()