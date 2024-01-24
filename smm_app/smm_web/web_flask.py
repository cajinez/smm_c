from flask import Flask
from flask import render_template
from flask import Response
import cv2  # OpenCV para tareas de visión por computadora
from deepface import DeepFace  # Biblioteca DeepFace para análisis facial
import numpy as np

from flask import request
import atexit

app = Flask(__name__)

# myrtmp_addr = "rtmp://localhost:1935/live/stream"
# cap = cv2.VideoCapture(myrtmp_addr)

cap = None

# cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) en windows

# Cargar el modelo preentrenado para la detección de emociones
model_emotion = DeepFace.build_model("Emotion")
model_gender = DeepFace.build_model("Gender")
model_age = DeepFace.build_model("Age")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
    "haarcascade_frontalface_default.xml")
# Definir las etiquetas de emociones
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
gender_labels = ['Male' , 'Female']

def generate():
    frame_count = 0
    while True:
        if cap is None:
            continue
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count % frame_skip == 0:
                # Convertir el frame a escala de grises
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detectar rostros en el frame
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y , w, h) in faces:
                    # Extraer la Región de Interés (ROI) del rostro
                    face_roi = gray_frame[y:y + h, x:x + w]

                    # Redimensionar la ROI del rostro para que coincida con la forma de entrada del modelo
                    resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

                    # Normalizar la imagen del rostro redimensionado
                    normalized_face = resized_face / 255.0

                    # Reorganizar la imagen para que coincida con la forma de entrada del modelo
                    reshaped_face = normalized_face.reshape(1, 48, 48, 1)

                    # Predecir emociones utilizando el modelo preentrenado
                    preds_emotion = model_emotion.predict(reshaped_face)[0]
                    emotion_idx = preds_emotion.argmax()
                    emotion = emotion_labels[emotion_idx]


                    # Extraer la Región de Interés (ROI) del rostro
                    face_roi = frame[y:y + h, x:x + w]

                    # Redimensionar la ROI del rostro para que coincida con la forma de entrada del modelo
                    resized_face = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)

                    # Normalizar la imagen del rostro redimensionado
                    normalized_face = resized_face / 255.0

                    # Reorganizar la imagen para que coincida con la forma de entrada del modelo
                    reshaped_face = normalized_face.reshape(1, 224, 224, 3)

                    # Predecir genero utilizando el modelo preentrenado
                    #roi_color=frame[y:y+h,x:x+w]
                    #roi_color=cv2.resize(roi_color,(224,224),interpolation=cv2.INTER_AREA)

                    #Predecir el género usando el modelo preentrenado
                    gender_predict = model_gender.predict(reshaped_face.reshape(-1,224,224,3))
                    gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
                    gender_label=gender_labels[gender_predict[0]] 




                    # Predecir edad utilizando el modelo preentrenado
                    age_predict = model_age.predict(reshaped_face)
                    # Obtener el índice de la clase con la probabilidad más alta
                    predicted_age_index = np.argmax(age_predict)
                    # Convertir el índice en la edad predicha
                    age = predicted_age_index + 1  # Sumar 1 porque los índices comienzan desde 0



                    # Dibujar un rectángulo alrededor del rostro y etiquetar con la emoción predicha, el género y la edad
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, gender_label, (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(frame, "Age="+str(age), (x+h,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/set_stream_key', methods=['POST'])
def set_stream_key():
    global cap
    global myrtmp_addr
    stream_key = request.form['stream_key']
    myrtmp_addr = f"rtmp://localhost:1935/live/{stream_key}"
    if cap is not None:
        cap.release()  # Liberar el objeto VideoCapture antes de crear uno nuevo
    try:
        cap = cv2.VideoCapture(myrtmp_addr)
    except cv2.error as e:
        print(f"Error al abrir el stream: {e}")
        cap = None
    return render_template("index.html")

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global cap
    if cap is not None:
        cap.release()  # Liberar el objeto VideoCapture antes de crear uno nuevo
    try:
        cap = cv2.VideoCapture(0)  # Usar la cámara
    except cv2.error as e:
        print(f"Error al abrir la cámara: {e}")
        cap = None
    return render_template("index.html")

frame_skip = 1  # Valor por defecto

@app.route('/set_frame_skip', methods=['POST'])
def set_frame_skip():
    global frame_skip
    frame_skip = int(request.form['frame_skip'])
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')  #La opcion de host permite que sea visible desde el exterior

def close_cap():
    cap.release()

atexit.register(close_cap)