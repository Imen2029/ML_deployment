# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 08:06:40 2022

@author: imen2
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
#import request
import json
from urllib.request import urlopen



API_URL='http://127.0.0.1:5000/'

# LOAD DATA
@st.cache(allow_output_mutation=True)
def load_data(url):
    req = requests.get(url)
    content = json.loads(req.content.decode('utf-8'))
    return pd.DataFrame(content['data'])

def main():
    data_load_state = st.text('Loading data...')
    response =requests.get("http://127.0.0.1:5000/")
    print(response.json())
    #infos = load_data(API_URL+'')
    #print(infos)
    #json_url = urlopen(API_URL)
    #API_data = json.loads(json_url.read())
    #print(API_data)
        
        
if __name__== '__main__':
    main()