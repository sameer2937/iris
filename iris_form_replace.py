# from re import S
from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

linear_model=pickle.load(open('iris_model.pkl','rb'))
columns_list=pickle.load(open('columns.obj','rb'))

@app.route('/')
def welcome():
    return 'welcome to IRIS page'

@app.route('/sepallengthprediction')
def prediction():
    SepalWidthCm = request.form.get('SepalWidthCm')
    PetalLengthCm = request.form.get('PetalLengthCm')
    PetalWidthCm = request.form.get('PetalWidthCm')
    Species = request.form.get('Species')

    if Species=='Iris-setosa':
        Species=0
    elif Species =='Iris-versicolor':
        Species=1
    elif Species == 'Iris-virginica':
        Species=2

    d={'SepalWidthCm':[SepalWidthCm],'PetalLengthCm':[PetalLengthCm],'PetalWidthCm':[PetalWidthCm],'Species':[Species]}

    test_df=pd.DataFrame(data=d)
    
    prediction=linear_model.predict(test_df)

    return jsonify({'prediction':prediction[0],
                    'SepalWidthCm':SepalWidthCm,
                     'PetalWidthCm':PetalWidthCm,
                      'Species':Species})

if __name__ == '__main__':
    app.run()