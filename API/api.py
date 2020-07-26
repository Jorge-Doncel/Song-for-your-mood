from src.app import app
from pymongo import MongoClient
from src.config import DBURL
import os
from flask import Flask, request, render_template,url_for, send_from_directory, redirect
from src.config import PORT
from bson.json_util import dumps
from faceRecognition import getFace, openImageAndDetectFaces, new_size
import json
from keras.models import model_from_json
import numpy as np

client = MongoClient(DBURL)
print(f"connected to db {DBURL}")
db = client.get_default_database()


app = Flask(__name__, template_folder='template')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    return redirect(url_for('msn', name=filename))

with open('model.json','r') as f:
    model_json = json.load(f)
model = model_from_json(model_json)
model.load_weights('my_model.h5')

@app.route("/message/<name>")
def msn(name):
    PIC = openImageAndDetectFaces(f"images/{name}")

    PIC = np.expand_dims(PIC,axis=0).reshape(np.expand_dims(PIC,axis=0).shape[0], 48, 48, 1)
    pred2 = model.predict(PIC)[0]
    return send_from_directory("images", name)

if __name__ == "__main__":
    app.run("0.0.0.0", PORT, debug=True)