import tkinter as tk
from tkinter import Label
import cv2
import threading
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('emotion_detection_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Detection")

        self.video_label = Label(root)
        self.video_label.pack()

        self.start_button = tk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.stop_button = tk.Button(root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.cap = None
        self.running = False
        self.thread = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.thread = threading.Thread(target=self.show_frame)
        self.thread.start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    def show_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                # Emotion prediction
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                prediction = model.predict(roi)[0]
                label = emotion_labels[np.argmax(prediction)]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

root = tk.Tk()
app = EmotionRecognitionApp(root)
root.mainloop()
