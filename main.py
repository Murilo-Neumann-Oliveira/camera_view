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

from ultralytics import YOLO


load_dotenv()

app = Flask(__name__)


# =====================================================
# CAMINHOS
# =====================================================

caminho_dados = os.getenv("dados")
caminho_modelo = os.getenv("modelo")
caminho_nuvens = os.getenv("nuvens")

print(f"Caminho dados: {caminho_dados}")
print(f"Caminho modelo: {caminho_modelo}")
print(f"Caminho Nuvens: {caminho_nuvens}")


# =====================================================
# YOLO SEGMENTAÇÃO NUVENS
# =====================================================

modelo_yolo = YOLO(
    caminho_nuvens
)



# =====================================================
# CALIBRAÇÃO CAMERA
# =====================================================


dados = np.load(
    caminho_dados
)


cameraMatrix = dados["cameraMatrix"]

distCoeffs = dados["distCoeffs"]




# =====================================================
# MODELO SOL
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
# POSIÇÃO SOLAR
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
        [[az,ze]]
    )



    px = int(previsao[0][0])
    py = int(previsao[0][1])


    return px,py




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

        self.cap=cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG
        )


        self.frame=None

        self.lock=threading.Lock()


        threading.Thread(
            target=self.update,
            daemon=True
        ).start()



    def update(self):

        while True:


            ret,frame=self.cap.read()


            if ret:

                with self.lock:

                    self.frame=frame


            else:

                time.sleep(1)




    def get_frame(self):

        with self.lock:

            if self.frame is None:

                return None


            return self.frame.copy()




camera=Camera(RTSP)





# =====================================================
# YOLO SEGMENTAÇÃO
# =====================================================


def segmentar_nuvens(frame):


    resultados = modelo_yolo(
        frame,
        conf=0.05,
        verbose=False
    )



    for r in resultados:


        if r.masks is None:

            continue



        masks = r.masks.data.cpu().numpy()



        for mask in masks:


            mask=cv2.resize(
                mask,
                (
                frame.shape[1],
                frame.shape[0]
                )
            )



            mask=(mask*255).astype(
                np.uint8
            )



            overlay=np.zeros_like(frame)



            overlay[:,:,0]=255



            frame[
                mask>100
            ] = (
                0.6*frame[
                    mask>100
                ]
                +
                0.4*overlay[
                    mask>100
                ]
            )


    return frame





# =====================================================
# STREAM
# =====================================================


def generate_frames():


    while True:



        frame=camera.get_frame()



        if frame is None:

            continue




        # ==============================
        # UNDISTORT
        # ==============================


        frame=cv2.undistort(
            frame,
            cameraMatrix,
            distCoeffs
        )



        # ==============================
        # YOLO NUVENS
        # ==============================


        frame=segmentar_nuvens(
            frame
        )



        # ==============================
        # TEMPO
        # ==============================


        timestamp=pd.Timestamp.now(
            tz="America/Sao_Paulo"
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )




        # ==============================
        # SOL
        # ==============================


        px,py=solar_pixel(
            timestamp
        )



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
            1,
            (0,255,0),
            2
        )



        cv2.putText(
            frame,
            f"Sol: {px},{py}",
            (50,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,0),
            2
        )




        # ==============================
        # JPEG
        # ==============================


        ret,buffer=cv2.imencode(
            ".jpg",
            frame
        )



        if ret:


            yield(
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            buffer.tobytes()
            +
            b"\r\n"
            )





# =====================================================
# WEB
# =====================================================


@app.route("/")
def index():

    return """

<html>

<body style="background:black;text-align:center">


<h1 style="color:white">
☀️ YOLO + Rastreamento Solar
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





if __name__=="__main__":


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )