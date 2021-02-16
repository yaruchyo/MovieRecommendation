from flask import Flask, request

app = Flask(__name__)


from app import views

app.config['SECRET_KEY'] = 'aY00yME0Jj6rt1aPBMTYBTg86GSgQkqY'