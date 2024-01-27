import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sounddevice as sd
import threading

from handtracking import draw_landmarks

frequency = 440.0

def synth():
    fs = 44100

    while True:
        print('hi')
        wave = np.sin(2 * np.pi * frequency * np.arange(fs) / fs)

        sd.play(wave, fs)
        sd.wait()

def webcam():
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = vid.read()

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)

        image =  mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.flip(frame, 1))
        detection_result = detector.detect(image)

        annotated_image = draw_landmarks(image.numpy_view(), detection_result)
        cv2.imshow('', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release() 
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    synth_thread = threading.Thread(target=synth)
    webcam_thread = threading.Thread(target=webcam)

    synth_thread.start()
    webcam_thread.start()

    synth_thread.join()
    webcam_thread.join()