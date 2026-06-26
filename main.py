from flask import Flask, Response
import cv2
import numpy as np
import pandas as pd
import pvlib
import joblib
import threading
import time
import os
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)

caminho_dados = os.getenv("dados")

caminho_modelo = os.getenv("modelo")

print(f"Caminho dados: {caminho_dados}")
print(f"Caminho modelo: {caminho_modelo}")
# =====================================================
# CALIBRAÇÃO DA CÂMERA
# =====================================================

dados = np.load(
    caminho_dados
)

cameraMatrix = dados["cameraMatrix"]
distCoeffs = dados["distCoeffs"]



# =====================================================
# MODELO DE RASTREIO DO SOL
# =====================================================

modelo = joblib.load(
    caminho_modelo
)


LOCAL = pvlib.location.Location(
    -29.713,
    -53.716,
    tz="America/Sao_Paulo"
)



# =====================================================
# FUNÇÃO POSIÇÃO SOLAR
# =====================================================

def solar_pixel(timestamp):

    tempo = pd.to_datetime(
        [timestamp]
    )


    pos = LOCAL.get_solarposition(
        tempo
    )


    az = pos["azimuth"].values[0]

    ze = pos["apparent_zenith"].values[0]


    previsao = modelo.predict(
        [[az, ze]]
    )


    px = int(previsao[0][0])
    py = int(previsao[0][1])


    return px, py




# =====================================================
# CAMERA RTSP
# =====================================================

os.environ[
    "OPENCV_FFMPEG_CAPTURE_OPTIONS"
] = "rtsp_transport;tcp"



RTSP = (
    "rtsp://admin:admin123@192.168.100.23:554/"
    "cam/realmonitor?channel=1&subtype=0"
)



class Camera:


    def __init__(self,url):

        self.cap = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG
        )


        self.frame = None

        self.lock = threading.Lock()


        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )


        self.thread.start()



    def update(self):

        while True:


            ret, frame = self.cap.read()


            if ret:


                with self.lock:

                    self.frame = frame


            else:

                print(
                    "Erro lendo camera"
                )

                time.sleep(1)



    def get_frame(self):

        with self.lock:

            if self.frame is None:

                return None


            return self.frame.copy()



camera = Camera(RTSP)




# =====================================================
# STREAM VIDEO
# =====================================================

def generate_frames():


    while True:


        frame = camera.get_frame()



        if frame is None:

            continue



        # ---------------------------------
        # CORREÇÃO DA DISTORÇÃO
        # ---------------------------------

        frame = cv2.undistort(
            frame,
            cameraMatrix,
            distCoeffs
        )



        # ---------------------------------
        # TEMPO ATUAL
        # ---------------------------------

        timestamp = pd.Timestamp.now(
            tz="America/Sao_Paulo"
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )



        # ---------------------------------
        # POSIÇÃO DO SOL
        # ---------------------------------

        px, py = solar_pixel(
            timestamp
        )



        # ---------------------------------
        # DESENHO
        # ---------------------------------

        cv2.circle(
            frame,
            (px,py),
            250,
            (0,0,255),
            3
        )


        cv2.circle(
            frame,
            (px,py),
            6,
            (0,255,255),
            -1
        )



        cv2.putText(
            frame,
            timestamp,
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,255,0),
            2
        )



        cv2.putText(
            frame,
            f"Sol: ({px},{py})",
            (50,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,0),
            2
        )



        # ---------------------------------
        # JPEG
        # ---------------------------------

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if not ret:

            continue



        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            buffer.tobytes()
            +
            b"\r\n"
        )





# =====================================================
# PAGINA WEB
# =====================================================


@app.route("/")
def index():

    return """

    <html>

    <head>

    <title>
    Rastreamento Solar
    </title>

    </head>


    <body style="background:black;text-align:center;">


    <h1 style="color:white;">
    ☀️ Rastreamento Solar em Tempo Real
    </h1>


    <img src="/video" width="90%">


    </body>

    </html>

    """




@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )





# =====================================================
# START
# =====================================================

if __name__ == "__main__":


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )