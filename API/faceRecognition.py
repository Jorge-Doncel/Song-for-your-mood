import pandas as pd
import cv2
import numpy as np
from numpy import asarray
from PIL import Image
import json
from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt
import face_recognition

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def openImageAndDetectFaces(path):
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)
    try:
        print("I found {} face(s) in this photograph.".format(len(face_locations)))
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            pil_image = Image.fromarray(face_image)
            plt.imshow(pil_image)
            return new_size(pil_image)
    except ValueError as e:
        print(f"No face found")
        
        
def new_size(img):
    size=(48,48)
    convert_from = img.resize(size)
    face=asarray(convert_from)/255
    return face