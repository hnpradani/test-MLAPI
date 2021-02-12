import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

#load model
model = pickle.load(open('model_iris.sav', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = [np.array([x for x in request.form.values()])]
    prediction = model.predict(final_features)

    return render_template('home.html', prediction_text='The variant of flower is ... {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)