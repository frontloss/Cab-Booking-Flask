# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:35:19 2020

@author: Abhinav Kumar
"""
#Importing Libraries
import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.preprocessing import LabelEncoder 
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

#Reading Train-Test data
train = pd.read_csv(r'C:\Edureka_AI_ML_course\Mid-program project 1\Dataset\train.csv')
train_label = pd.read_csv(r'C:\Edureka_AI_ML_course\Mid-program project 1\Dataset\train_label.csv')
test = pd.read_csv(r'C:\Edureka_AI_ML_course\Mid-program project 1\Dataset\test.csv')
test_label = pd.read_csv(r'C:\Edureka_AI_ML_course\Mid-program project 1\Dataset\test_label.csv')

train_labels_list = [504] + list(train_label['504'])
if 'Cab Bookings' not in list(train.columns):
    train['Cab Bookings'] = train_labels_list 

test_labels_list = [256] + list(test_label['256'])
if 'Cab Bookings' not in list(test.columns):
    test['Cab Bookings'] = test_labels_list 

#Weekday, Month conversion functions
def weekday_val(weekday):
    weekday_dict = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    return weekday_dict[weekday]
def month_val(month):
    month_dict = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,
                  'October':10,'November':11,'December':12}
    return month_dict[month]

#In train data
train['datetime'] = pd.to_datetime(train['datetime'])
datetime_list = ['WeekDay','Day','Time','Month','Year']
train['WeekDay'] = train['datetime'].apply(lambda x: x.strftime('%A'))
train['Day'] = train['datetime'].apply(lambda x: x.strftime('%d'))
train['Time'] = train['datetime'].apply(lambda x: x.strftime('%X'))
train['Month'] = train['datetime'].apply(lambda x: x.strftime('%B'))
train['Year'] = train['datetime'].apply(lambda x: x.strftime('%Y'))
train = train.drop('datetime',axis=1)

#In test data
test['datetime'] = pd.to_datetime(test['datetime'])
datetime_list = ['WeekDay','Day','Time','Month','Year']
test['WeekDay'] = test['datetime'].apply(lambda x: x.strftime('%A'))
test['Day'] = test['datetime'].apply(lambda x: x.strftime('%d'))
test['Time'] = test['datetime'].apply(lambda x: x.strftime('%X'))
test['Month'] = test['datetime'].apply(lambda x: x.strftime('%B'))
test['Year'] = test['datetime'].apply(lambda x: x.strftime('%Y'))
test = test.drop('datetime',axis=1)

new_train = train.copy()
new_test = test.copy()

#For train data
new_train['Month'] = new_train['Month'].apply(lambda x:month_val(x))
new_train['WeekDay'] = new_train['WeekDay'].apply(lambda x:weekday_val(x))
new_train['Hour'] = new_train['Time'].apply(lambda x:int(x[:2]))
new_train['weather'] = LabelEncoder().fit_transform(new_train['weather'])
new_train['season'] = LabelEncoder().fit_transform(new_train['season'])

#For test data
new_test['Month'] = new_test['Month'].apply(lambda x:month_val(x))
new_test['WeekDay'] = new_test['WeekDay'].apply(lambda x:weekday_val(x))
new_test['Hour'] = new_test['Time'].apply(lambda x:int(x[:2]))
new_test['weather'] = LabelEncoder().fit_transform(new_test['weather'])
new_test['season'] = LabelEncoder().fit_transform(new_test['season'])

#Feature Selection
X_train = new_train[['Hour','Month','temp','humidity','windspeed','WeekDay',
                     'Year']]
Y_train = new_train['Cab Bookings']
X_test = new_test[['Hour','Month','temp','humidity','windspeed','WeekDay',
                     'Year']]
Y_test = new_test['Cab Bookings']
rf = RandomForestRegressor(random_state=0,n_estimators=100,max_features=6)
rf.fit(X_train,Y_train)

#Displaying R2-scores 
Y_test_pred = np.round_(rf.predict(X_test))
Y_train_pred = np.round_(rf.predict(X_train))
print("Test R2 Score",r2_score(Y_test, Y_test_pred))
print("Train R2 Score",r2_score(Y_train,Y_train_pred))

#Saving model
pickle.dump(rf,open('random_forest.pkl','wb'))
