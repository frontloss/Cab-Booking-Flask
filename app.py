# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 00:02:48 2020

@author: Abhinav Kumar
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('random_forest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    def weekday_month_val(val):
        comb_dict = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
                     'Friday':4,'Saturday':5,'Sunday':6,
                     'January':1,'February':2,'March':3,'April':4,'May':5,
                     'June':6,'July':7,'August':8,'September':9,
                     'October':10,'November':11,'December':12}
        return comb_dict[val]
    features = []
    for x in request.form.values():
        try:
            float_val = float(x)
            features.append(float_val)
        except Exception as _:
            val = weekday_month_val[x]
            features.append(val)
            continue
    
    features.append(2011)
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    pred_2011 = round(prediction[0])
    features.remove(2011)
    
    features.append(2012)
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    pred_2012 = round(prediction[0])
    
    avg_pred = round((pred_2011+pred_2012)/2)
    return render_template('index.html', prediction_2011='{}'.format(pred_2011),
                           prediction_2012='{}'.format(pred_2012),
                           avg_prediction='{}'.format(avg_pred))
  
if __name__ == "__main__":
    app.run(debug=True)
