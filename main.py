from flask import Flask, Response
import cv2
import numpy as np
import pandas as pd
import pvlib
import joblib

app = Flask(__name__)

# ======================
# CALIBRAÇÃO
# ======================
dados = np.load(r"C:\Mestrado\03_Processamento_Imagens_PV\03_Calibracao_Camera\calibration_camera.npz")
cameraMatrix = dados['cameraMatrix']
distCoeffs = dados['distCoeffs']

# ======================
# MODELO SOLAR
# ======================
modelo = joblib.load(r'C:\Mestrado\04_Modelos_Machine_Learning\02_Rastreio_Sol\modelo_rastreio_sol.pkl')

LOCAL = pvlib.location.Location(-29.713, -53.716, tz='America/Sao_Paulo')

def solar_pixel(timestamp):
    t = pd.to_datetime([timestamp])
    pos = LOCAL.get_solarposition(t)

    az = pos['azimuth'].values[0]
    ze = pos['apparent_zenith'].values[0]

    px, py = modelo.predict([[az, ze]])[0]
    return int(px), int(py)

# ======================
# RTSP CAMERA
# ======================
RTSP = "rtsp://admin:admin123@192.168.100.23:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)

# ======================
# STREAMING FRAME
# ======================
def generate_frames():
    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        # UNDISTORT
        frame = cv2.undistort(frame, cameraMatrix, distCoeffs)

        # SOL
        timestamp = pd.Timestamp.now(tz='America/Sao_Paulo')
        px, py = solar_pixel(str(timestamp))

        # DRAW
        cv2.circle(frame, (px, py), 250, (0, 0, 255), 3)
        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)

        # ENCODE JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ======================
# ROUTE WEB
# ======================
@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Rastreamento Solar</title>
        </head>
        <body style="background:black; text-align:center;">
            <h1 style="color:white;">☀️ Rastreamento Solar em Tempo Real</h1>
            <img src="/video" width="80%">
        </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ======================
# START SERVER
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)