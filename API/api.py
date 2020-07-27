from src.app import app
import os
from flask import Flask, request, render_template,url_for, send_from_directory, redirect
from src.config import PORT
from bson.json_util import dumps
from faceRecognition import getFace, openImageAndDetectFaces, new_size
import json
from keras.models import model_from_json
import numpy as np
import pandas as pd

df=pd.read_csv('../data/songs_clean.csv')
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

    return redirect(url_for('answer', name=filename))

with open('fer.json','r') as f:
    model_json = json.load(f)
model = model_from_json(model_json)
model.load_weights('fer_model.h5')

@app.route("/answer/<name>")
def answer(name):
    PIC = openImageAndDetectFaces(f"images/{name}")
    senti= ("angry", "disgust", "fear", "happy", "sadness", "surprise", "neutral")
    PIC = np.expand_dims(PIC,axis=0).reshape(np.expand_dims(PIC,axis=0).shape[0], 48, 48, 1)
    pred2 = model.predict(PIC)[0]
    feeling=senti[max(range(len(pred2)), key = lambda x: pred2[x])]
    feeling_song= df[df["senti"]==feeling].sample(n=1)
    song= feeling_song.iloc[0]["track_name"]
    artist= feeling_song.iloc[0]["track_artist"]
    album= feeling_song.iloc[0]["track_album_name"]
    playlist= feeling_song.sample(n=1).iloc[0]["playlist_name"]
    return dumps(f'Today you are {senti[max(range(len(pred2)), key = lambda x: pred2[x])]}. I recommend the song {song} by {artist} from the album  {album}. You can find in the Spotify playlist call {playlist}')
if __name__ == "__main__":
    app.run("0.0.0.0", PORT, debug=True)