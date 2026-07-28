from flask import Flask, Response, request
from camera import gen_frames, release_cameras

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <html>
      <body style="margin:0;background:#111;color:white;font-family:Arial;text-align:center;">
        <img id="video" src="/video_feed" style="width:100%;max-width:1000px;border:1px solid #444;">
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        release_cameras()