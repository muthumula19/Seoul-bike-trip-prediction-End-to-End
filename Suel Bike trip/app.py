import os
import numpy as np
import pandas as pd
from flask import Flask,jsonify,request,render_template
import flask
import xgboost as xgb
import Model
app=Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello World!'
@app.route('/index')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    columns_names=[   'Pmonth','Pday', 'Phour', 'Pmin', 'PDweek', 'Temp', 'Precip', 'Wind', 'Humid',\
       'Solar', 'Snow', 'GroundTemp', 'Dust']
    #model=pickle.load(open('model','rb'))
    value=request.form.to_dict() #convert json to dictonry

    val=list()
    for i in columns_names:
        try:
            val.append(float(value[i]))
        except:
            return render_template('index.html',results='Check the values you enterd ....')
    if not os.path.isfile('model.txt'):
        Model.train()
    model=xgb.Booster()  #xgb booster for load our model xgbregressor
    model.load_model('model.txt')  #Hrere we load model.txt
    test_data=[val]
    test=pd.DataFrame(test_data,columns=columns_names) #Create data frame
    test1=xgb.DMatrix(test) #Create Dmatrix for xgb
    output=model.predict(test1) #predict
    return render_template('index.html',results="Trip Duration {} min with deaviation of {} min".format(int(output[0]),int(output[0])/10))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    
    

    
