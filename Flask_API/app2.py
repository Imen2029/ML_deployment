# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 08:07:20 2022

@author: imen2
"""

import pickle
import pandas as pd
import numpy as np
from  lightgbm import  LGBMClassifier
from flask import Flask ,jsonify
import json

app=Flask(__name__)
app.config["DEBUG"] = True
#load the model
infile1=open('LGBMClassifier_f2score_is_unbalance.pkl','rb')
model=pickle.load(infile1)
#load the original data test
test=pd.read_csv('application_test.csv')

#load the test dataset after preprocessing and feature engineeing (categorical data is onehot encoded)
data_test=pd.read_csv('test_data.csv')
data_test.set_index('SK_ID_CURR' ,inplace=True)


data=data_test.drop(['TARGET'] ,axis= 1)
        
cols_infos = ['CODE_GENDER','NAME_FAMILY_STATUS', 
               'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
               'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_CONTRACT_TYPE', 
             ]
#test_sk=test.loc[test.SK_ID_CURR.isin(SK_ID_CURR_LIST)]#to have access to the initial categorical values 

infos=data[cols_infos]
#Home page
@app.route('/')
def home():
    return "API pour le Dashboard \'Prêt à dépenser\' "

# GENERAL INFORMATIONS
@app.route('/info', methods=['GET'])
def get_infos():        
    # Converting the pd.DataFrame to JSON
    info_json = json.loads(infos.to_json())
    return jsonify({ 'data' : info_json})    

# GENERAL INFORMATIONS ON SELECTED CLIENT
@app.route('/info/<int:id_client>', methods=['GET'])
def get_info_id(id_client):
    
    info_client_select = infos.loc[id_client,:]
    
    # Converting the pd.Series to JSON
    data_client_json = json.loads(info_client_select.to_json())

    return jsonify({ 'data' : data_client_json})

# PREDICTIONS ON SELECTED CLIENT
@app.route('/prediction/<int:id_client>', methods=['GET'])
def get_data_pred(id_client):
    data_client_select=data.loc[id_client:id_client]    
    # Converting the pd.Series to JSON
    data_client_json=json.loads(data_client_select.to_json())
    # Make prediction
    client_proba=model.predict_proba(data_client_select)[:, 1]
    if client_proba<0.525:
        decision='non default customer'
    else:
        decision='default customer'


    return jsonify({ 'data' : data_client_json,
                    'score' : client_proba,
                    'decision':decision})


app.run()