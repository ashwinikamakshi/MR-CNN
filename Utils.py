import os
import numpy as np
import cv2 as cv2
from pandas import *

def load_data(path):
    org=[]
    data = read_csv(path+'Five_Labels.csv')
    # converting column data to list
    grade = data['Grade'].tolist()
    labels = np.array(grade) #converting list to array

    for filename in os.listdir(path+'clahe_five'):
        image = cv2.imread(path+'clahe_five\\'+filename)
        image=cv2.resize(image,(224,224))
        org.append(image)

    x, y = (org,labels)

    x = np.array(x, np.float32)
    x = x / 255.

    return x, y

