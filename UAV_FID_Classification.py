# -*- coding: utf-8 -*-

# from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import LinearRegression
# from tensorflow.keras.losses import MeanSquaredLogarithmicError
from keras.callbacks import EarlyStopping
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

train_path = '/content/drive/MyDrive/uav/obj_train_data'

OUTLIER_PECENTAGE = 11

REPAIRED_DATA = 2

x_train=[]

for img in os.listdir(train_path):

    # sub_path=train_path+"/"+folder

    # for img in os.listdir(train_path):

        image_path=train_path+"/"+img

        img_arr=cv2.imread(image_path)

        if(type(img_arr) == type(None)):
          pass
        else:
          img_arr=cv2.resize(img_arr,(224,224))

          img_arr = img_arr.reshape(img_arr.shape[0] * img_arr.shape[1]* img_arr.shape[2])

          img_arr=img_arr/255.0

          x_train.append(img_arr)

train_x=np.array(x_train)
# train_x = pd.DataFrame(x_train).T

total_entry = int(train_x.shape[0])

# train_x=train_x/255.0

random_indices_train = np.random.choice(total_entry, int(total_entry/100*OUTLIER_PECENTAGE), replace = False)

for index in random_indices_train:
# # #   for j in range(train_x.shape[1]):
    sigma_RH = 0.1* train_x[13][index] 
    noise_RH = random.gauss(0, sigma_RH)
    train_x[13][index] = train_x[13][index]+noise_RH
# # #     # train_x[index][j] = 0

# Label the outlier and inlier
y = np.ones(train_x.shape[0])
y_fix = y
for i in random_indices_train:
    y[i] = 0

# # # FID for repairing data
def FID_repaired(working_list,total_number_fix):
    # working_list = x1
    
    # index = []
    # for i, j in enumerate(working_list):
    #     if j == 'NaN':
    #         index.append(i)
    
    # count the number of NaN/compromised point
    # p1 = working_list.count('NaN')
    # print(p1)
    
    t = total_number_fix
    
    # Select the min & max from list
    # # working_list.remove('NaN')
    # count=0
    # for index_pos in index:
    #     working_list.pop(index_pos-count)
    #     count+=1
    
    # find mean of all observed values
    mean = np.mean(working_list)
    
    #find min value
    a = min(working_list)
    
    #find max value
    b = max(working_list)
    
    # Calculate h = (b-a)/t
    h = (b-a)/t
    
    # Calculate the discrete universe U using u = (a + (s-1) x h + a + s x h)/2, s=1,2,3
    U = []
    for s in range(1,t+1):
        u = (a + (s-1) * h + a + s * h)/2
        U.append(u)
    
    # print(U)
        
    # Calculating the missing values
    M = []    
    for u in U:
        # print(U)
        
        # Compute the contribution weight (micro) of each observed element x_i
        
        contribution_weight_list = []
        
        for i in working_list:
            if abs(i-u) <= h:
                temp = 1-(abs(i-u)/h)
            else:
                temp = 0
            contribution_weight_list.append(temp)
        
        # Calculate the sum of x_i to u1:
        sum_contribution_weight_list = sum(contribution_weight_list)
        # print(sum_contribution_weight_list)
        
        # Calculate the contribution of an observed data x_i
        sum_contribution_observed_data = []
        
        for num1, num2 in zip(working_list, contribution_weight_list):
        	sum_contribution_observed_data.append(num1 * num2)
        
        sum_contribution_observed_data = sum(sum_contribution_observed_data)
        # print(sum_contribution_observed_data)
        
        # Calculate the missing values in x_i
        if sum_contribution_weight_list == 0:
            m = mean
        else:
            m = sum_contribution_observed_data/sum_contribution_weight_list
        
        M.append(m)
    
    # print('The values:',M)
    # print('The index position:',index)
    return M

# # Repairing data
data = train_x[13]

# # # # Determine the outlier for repairing
# index_outlier = []

# y = df['label2']
# count_outlier = 0
# for i, j in enumerate(y):
#     if j == 0.0:
#         index_outlier.append(i)
#         count_outlier+=1
#     if count_outlier == PERCENTAGE_REPAIRED: break

# Determine the compromised data
# for ind in random_indices_train:
#     data[ind] = 'NaN'

# Recover compromised data
working_list = data.tolist()

results = FID_repaired(working_list,int(total_entry/100*REPAIRED_DATA))

# # # Update the predicted data into dataset
total_repaired = int(total_entry/100*REPAIRED_DATA)
random_indices_train = random_indices_train[:total_repaired]

pos = 0
for index_pos in results:
  train_x[13][random_indices_train[pos]] = index_pos
  # df_1['dur'] = results[1][pos]
  # df_1['label2'] = 1

  # df.loc[index_pos] = df_1
  pos+=1

for i in random_indices_train:
    y[i] = 1

# Splitting the for testing and validating dataset
x_train, x_test, y_train, y_test = train_test_split(train_x,y, test_size=0.33,stratify=y)

# # Random Forest
forest = RandomForestClassifier(n_estimators = 10, random_state = 0)
forest.fit(x_train, y_train)
preds = forest.predict(x_test)

# Metrics
print('accuracy_score RF: ', round((accuracy_score(y_test,preds)*100),2))
# print('f1_score: ', round(f1_score(y_test,preds),6))
# print('precision_score: ', round(precision_score(y_test,preds),6))
# print('recall_score: ', round(recall_score(y_test,preds),6))
# tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
# specificity = tn/(tn+fp)
# print('Specificity : ', round(specificity,6))

# # # KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train, y_train)

preds = neigh.predict(x_test)

# Metrics
print('accuracy_score: ', round((accuracy_score(y_test,preds)*100),2))
# print('f1_score: ', round(f1_score(y_test,preds),6))
# print('precision_score: ', round(precision_score(y_test,preds),6))
# print('recall_score: ', round(recall_score(y_test,preds),6))
# tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
# specificity = tn/(tn+fp)
# print('Specificity : ', round(specificity,6))

#### DecisionTreeClassifier #####

clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train, y_train)

preds = clf.predict(x_test)

# Metrics
print('accuracy_score: ', round((accuracy_score(y_test,preds)*100),2))
# print('f1_score: ', round(f1_score(y_test,preds),6))
# print('precision_score: ', round(precision_score(y_test,preds),6))
# print('recall_score: ', round(recall_score(y_test,preds),6))
# tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
# specificity = tn/(tn+fp)
# print('Specificity : ', round(specificity,6))

##### SVC #####

svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

accuracy_test = np.mean(y_test == y_pred) * 100
print('Accuracy for the test dataset: ', round(accuracy_test,2))
