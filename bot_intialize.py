import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template
from app import prediction

app = Flask(__name__, template_folder="templates")

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/crypto', methods = ['POST'])
def predict():
     if request.method == 'POST':
        crypto = request.form.values()
        crypto = list(map(str, crypto))[0]
        print("HELLO")
        pred = prediction(crypto)
        print(pred)
        return  flask.render_template('predict.html',data=pred)

if __name__ == '__main__':
      app.run(host='127.0.0.1',port=5000, debug=True)