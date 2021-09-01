from flask import Flask, render_template, request, redirect, url_for
from flask_cors import cross_origin
import pickle
import numpy as np
from model import Model
from prediction_transformer import Predictor_Data_Transformer
from logger import Logger

app = Flask(__name__)

@cross_origin
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@cross_origin
@app.route('/pandas_profiling')
def profile():
    return render_template('profile.html')

@cross_origin
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    #model = Model('ai4i2020.csv').best_model()
    model = pickle.load(open('model.sav', 'rb'))
    if request.method == 'POST':
        try:
            Type = request.form.get('Type')
            process_temp = float(request.form.get('Process Temp'))
            rpm = float(request.form.get('RPM'))
            torque = float(request.form.get('Torque'))
            wear = float(request.form.get('wear'))
            failure = int(request.form.get('Failure'))
            twf = int(request.form.get('TWF'))
            hdf = int(request.form.get('HDF'))
            pwf = int(request.form.get('PWF'))
            osf = int(request.form.get('OSF'))
            rnf = int(request.form.get('RNF'))
            data = Predictor_Data_Transformer(Type, process_temp, rpm, torque, wear, failure, twf, hdf, pwf, osf, rnf).data()
            predictor = round(model.predict(data)[0], 2)
            return render_template('prediction.html', temp=predictor)
        except Exception as e:
            Logger('test.log').logger('ERROR', str(e))
            return redirect(url_for('home'))


if __name__ == '__main__':
    app.run()