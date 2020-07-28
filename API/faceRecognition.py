import pandas as pd
import cv2
import numpy as np
from numpy import asarray
from PIL import Image
import json
from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

def getFace(img):
    img_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img_bw, 1.1, 4)

    # Ensure there is at least 1 face
    if len(faces) > 0:
        print("Face detected!")
        x,y,w,h = faces[0]
        return new_size(img_bw[y:y+h,x:x+w])
    else:
        raise ValueError("No face found")
        
        
def openImageAndDetectFaces(path):
    img = cv2.imread(path)
    plt.imshow(img)
    try:
        print(f"Detecting faces in {path}")
        face_patch = getFace(img)
        return face_patch
    except ValueError as e:
        print(f"Not found image in {path}")
        return None

def new_size(img):
    size=(48,48)
    convert_to = Image.fromarray(img)
    convert_from = convert_to.resize(size)
    face=asarray(convert_from)/255
    return face