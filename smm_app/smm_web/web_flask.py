from flask import Flask
from flask import render_template
from flask import Response
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) en windows
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 
    "haarcascade_frontalface_default.xml")

def generate():
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y , w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

@app.route("/")
def home():
    return render_template("index.html")


cap.release()