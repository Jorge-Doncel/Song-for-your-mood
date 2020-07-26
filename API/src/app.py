from flask import Flask
import os


app = Flask(__name__, template_folder='template')


APP_ROOT = os.path.dirname(os.path.abspath(__file__))