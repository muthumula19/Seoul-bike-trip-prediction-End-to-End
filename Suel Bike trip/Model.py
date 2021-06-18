import numpy as np 
import pandas as pd
import pickle
import os
#from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import xgboost as xgb
def train():
    data=pd.read_csv('data.csv',nrows=1000)    #import data for our model
    data=data[['Pmonth', 'Pday', 'Phour', 'Pmin', 'PDweek','Temp', 'Precip', 'Wind','Humid', \
                'Solar', 'Snow', 'GroundTemp', 'Dust','Duration']]
    "Convert the data into time series by using Pmonth,Pdat,Phour,Pmin"
    month=["%02d" % i for i in data['Pmonth']]
    day=["%02d" % i for i in data['Pday']]
    hour=["%02d" % i for i in data['Phour']]
    min=["%02d" % i for i in data['Pmin']]
    datetime=[]
    for m,d,h,min in zip(month,day,hour,min):
        date=str(m)+str(d)+str(h)+str(min)
        datetime.append(date)
    data['datetime']=datetime
    data['datetime']=pd.to_datetime(data.datetime,format='%m%d%H%M')
    data=data.resample('10T',on='datetime').mean().reset_index() #sample the time series data 10min intervel and take mean of samples
    data=data.dropna()#removing null values
    data=data.drop('datetime',axis=1) #reamove date time column from data
    y=data['Duration'] #Separetrate Dependent varible from data
    X=data.drop(['Duration'],axis=1) #Separate independent varible from data
    #Creating pipline for our data modeling
    xgb_model=XGBRegressor(n_estimators=50,max_depth=5,learning_rate=0.1,n_jobs=-1,random_state=42) #applying XGBRegressor to our data
    xgb_model.save_model('model1.txt')
    
def predict():
    columns_names=[   'Pmonth','Pday', 'Phour', 'Pmin', 'PDweek', 'Temp', 'Precip', 'Wind', 'Humid',
       'Solar', 'Snow', 'GroundTemp', 'Dust']
    test_data=[]
    for i in columns_names:
        v=float(input('Enter value of {} '.format(i)))  #taking input values from user to predict duration
        test_data.append(v)   
    test_data=[test_data]
    test_=pd.DataFrame(test_data,columns=columns_names)
    if not os.path.isfile('model.txt'):
        model=train()
        output=model.predict(test_)
    else:

        model=xgb.Booster()  #xgb booster for load our model xgbregressor
        model.load_model('model.txt')  #Hrere we load model.txt
        
        test=pd.DataFrame(test_data,columns=columns_names) #Create data frame
        test=xgb.DMatrix(test) #Create Dmatrix for xgb
        output=model.predict(test)
    print('Trip Duration',output)

def name():
    print('name')
