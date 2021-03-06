# -*- coding: utf-8 -*-
"""boneageTest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M6JutHkozmqLnkGygfwaeX2MY69z4eBG
"""
import scipy

# # 구글 코랩에서 드라이브 연동
# from google.colab import drive
# drive.mount('/content/drive')

from flask import Flask, jsonify #간단히 플라스크 서버를 만든다
import urllib.request

import tensorflow as tf
import pandas as pd #이미지 불러오기
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np

#개월->년
def changeYear(ARR) :
  NewARR = []
  for num in ARR :  
    year = num / 12
    remainder = year % 1
    
    if remainder <= 0.25 :
      remainder = 0
      NewARR.append(int(year) + remainder)
    elif remainder <= 0.75 :
      remainder = 0.5
      NewARR.append(int(year) + remainder)
    else :
      remainder = 1
      NewARR.append(int(year) + remainder)
  return NewARR

#성인예측키                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        퓿
def Height_prediction ( male, BA, current_H) :
    import pandas as pd
#    lms_df = pd.read_csv('/content/drive/Shareddrives/growthPrediction/machineLearning/2017_성장도표_데이터테이블_공개용_최종.csv')
    lms_df = pd.read_csv('2017_2.csv', encoding = "utf-8")

    # return lms_df

    month_age = np.round(BA * 12)
    if male == True:
        lms_index = month_age                                              
        L_18, M_18, S_18 = lms_df.iloc[226,3], lms_df.iloc[226,4], lms_df.iloc[226,5]
    elif male == False:
        lms_index = month_age 
        L_18, M_18, S_18 = lms_df.iloc[454,3], lms_df.iloc[454,4], lms_df.iloc[454,5]

    L,M,S = lms_df.iloc[lms_index,3], lms_df.iloc[lms_index,4],lms_df.iloc[lms_index,5]
    x = current_H
    
    Z = (((x/M)**L)-1)/(L*S)
    Z = round(Z,4)

    pred_height = M_18 * (1 + (L_18 * S_18 * Z)) ** (1 / L_18)
    pred_height = round(pred_height, 1)
    return pred_height

app = Flask(__name__)

import base64
import pymysql
import io
import PIL.Image as Image

@app.route('/')
def home():
    return 'home'

@app.route('/test')
def test():
  import tensorflow as tf
  model = tf.keras.models.load_model("./check_point_3")
  print("finish load!!!!!!!!!!!!!!!!!")
  return 'test'


        
    
if __name__ == '__main__':
    app.run(debug=False,host="127.0.0.2",port=5000)
