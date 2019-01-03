#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:54:36 2018

@author: sku014
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
#from lstm_functions import lstm_train_fkn
#from lstm_functions import load_data
#from lstm_functions import normalize_data
from lstm_functions import split
from keras.models import load_model
from keras import backend as K
#import os, datetime
import random
import pickle
import itertools #Eg for permutations to work


#%% Load variables
save_path = '/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/Save_folder/x_Experiments/Keras_lstm_feature_combinations10/'  #Office
#save_path = '/Users/sku014/Documents/MATLAB/MLDIF/save_folder/Keras_lstm_feature_combinations/'      #Laptop 
# Getting back the objects:
with open(save_path + 'variables.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loss_tr, loss_vl, D, Y, VOIid, time_scale, split_idx, exp_mat = pickle.load(f)

#%% Load saved models one by one and make +store the predictions
Yte_exp = np.zeros((12, D.shape[1], len(exp_mat)))
#yte_it = np.zeros((12, D.shape[1], split_idx.shape[1]))    #Ground truth, for debugging, can be removed later

t = np.arange(0,10,1)           #This is all features to be combined
experiment = 0                  #Experiment counter. To become 1023
#laptop_test = [1,2,3,10]

#Loop all number of features (numfeat): 1, 2, 3...10
for numfeat in range(1,11):
    print(numfeat)

    c = list(itertools.combinations(t, numfeat))     #c is a list of all possible combinations of numfeat elements from t.
    
    
    #Loop all list elements of c
    for el in range(0,len(c)):
#        print(el)
        print(experiment)
        
#        exp_mat.append([[laptop_test[itr]], c[el]])  # Store the number of features and current combination in the experiment matrix.
        
        D_extr = D[:,:,c[el]]       #D_extracted contains selected features only, as given in the list c.
#        Xtr, Xvl, Xte, Ytr, Yvl, Yte, IDtr, IDvl, IDte  = split(D_extr,Y,VOIid, split_idx[:,i])     #Extract training, validation and test data with only the current features, given in c[el]
   
    
        #Clear the last session from memory
        K.clear_session()
        
        #Load the model
        model = load_model(save_path + '/1-layer-lstm_feature_combination_experiment_' + str(experiment) + '.h5')
        
        #Obtain the current training and test set
        Xtr, Xvl, Xte, ytr, yvl, yte, IDtr, IDvl, IDte  = split(D_extr,Y,VOIid, split_idx[:,0])      #Not sure that IDtr, IDvl and IDte are needed. Check when everything is done. All is handled through split_idx and the split function
        
        #Make predictions and invert them
        #Ytr = model.predict(Xtr)
        Yte = model.predict(Xte)    #Model predictions: y-hat-te represented by: Yte
        
        #Ytr = Ytr.squeeze()
        Yte = Yte.squeeze()
        
        #ytr = ytr.squeeze()
        #yte = yte.squeeze()
        
        #Ytr = scaler.inverse_transform(Ytr)
        #Yte = scaler.inverse_transform(Yte)
        
        #ytr = scaler.inverse_transform(ytr)
        #yte = scaler.inverse_transform(yte)
        
        #Store the predictions from the i:th iteration
        Yte_exp[:,:,experiment] = Yte  
    #    yte_it[:,:,i] = yte     #Just for debugging/plotting below, can be removed later.
        experiment += 1

#%% Calculate the RMSE for all models

RMSE_mat = np.sqrt(((Yte_exp - yte) ** 2).mean(axis=1))

mean_RMSE = RMSE_mat.mean(axis=0)   #RMSE over all test mice in this particular shuffling
loss_vl_min = np.zeros(len(loss_vl))

for q in range(0,len(loss_vl)):
    loss_vl_min[q] = min(loss_vl[q])

#plt.plot(loss_vl_min, mean_RMSE, 'o') 
#plt.xlabel('Validation loss')
#plt.ylabel('RMSE_mean')
#%% For debugging. make some plots
#test_idx = split_idx[-12:,:]
#
##RMSE_mat = np.sqrt(((Yte_exp - yte) ** 2).mean(axis=1))
#  
##exp=2
#mouse_te = random.randint(0,Yte_exp.shape[0]-1)
#
#for exp in range(0,Yte_exp.shape[2]):
#    plt.figure(figsize=(10,6))
#    
#    
#    plt.subplot(2, 2, 3)
#    plt.plot(time_scale, yte[mouse_te,:])
#    plt.plot(time_scale, Yte_exp[mouse_te,:,exp])
#    
#    RMSE = np.sqrt(((Yte_exp[mouse_te,:,exp] - yte[mouse_te,:,0]) ** 2).mean())
#    
#    plt.title('0-60 min. Test Mouse ' + str(VOIid[test_idx[mouse_te,0]][0]) + ' RMSE = ' + str(round(RMSE,3)))
#    plt.legend(['LV+VC', 'prediction'], loc='upper right')
#    
#    plt.subplot(2, 2, 4)
#    plt.plot(time_scale[0:33], yte[mouse_te,0:33])
#    plt.plot(time_scale[0:33], Yte_exp[mouse_te,0:33,exp])
#    
#    plt.title('0-5 min. Test Mouse ' + str(VOIid[test_idx[mouse_te,0]][0]))
#    plt.legend(['LV+VC', 'prediction'], loc='upper right')
#    
#    plt.tight_layout()                  #Use tight layout
#    plt.show()
 
#np.sqrt(((np.squeeze(Yte_exp[0,:,3])-np.squeeze(yte[0,:,0])) **2).mean())

#%% Save variables to disk, since model prediction takes some time.

# Saving the objects:
with open(save_path + '/predictions.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Yte_exp, exp_mat, RMSE_mat], f)

#%% Save the results to a readable Matlab file
sio.savemat('/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/LSTM_feature_combinations_RMSE_mat10.mat', {'RMSE_mat':RMSE_mat})
sio.savemat('/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/LSTM_feature_combinations_exp_mat10.mat', {'exp_mat':exp_mat})
sio.savemat('/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/LSTM_feature_combinations_Yte_exp10.mat', {'Yte_exp':Yte_exp})
sio.savemat('/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/LSTM_feature_combinations_IDte10.mat', {'IDte':IDte})
sio.savemat('/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/LSTM_feature_combinations_split_idx10.mat', {'split_idx':split_idx})

    
#%% Load Yte_it in case things crashed!
    
##save_path = '/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/Save_folder/Keras_lstm/'  #Office
##save_path = '/Users/sku014/Downloads/Keras_lstm_100it/'      #Laptop 
## Getting back the objects:
#with open(save_path + 'predictions.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#    Yte_it = np.array(pickle.load(f))
#Yte_it = np.squeeze(Yte_it)

#%%  Continue to process
#
#loss_vl_min = np.zeros(1000)
## Store the minimum loss
##loss_vl_min = np.min(loss_vl, axis=0)          #Get the minimum loss. #Old method working with loss_vl as np.array
#for i in range(0,1000):                         #New method working with loss_vl as list.
#    loss_vl_min[i] = min(loss_vl[i])
#    
##loss_vl_min_idx = np.argmin(loss_vl, axis=0)   #The index of the minimum loss (not always the last!). Not really needed, just for curiosity
#
## Extract the indexes of the test set
#test_idx = split_idx[-12:,:]
#
##Reshape Yte_it to the same shape az test_idx
#Yte_it = np.transpose(Yte_it, (0,2,1))  #E.g. have the shape: (12,100,44)
#
##test2 = np.arange(0,68)
#
##FOR LOOP HERE ON INDEX TEST2 ABOVE!
##mouse_idx_it = list()
##mouse_idx_row = list()
##mouse_idx_loss = list()
##mouse_idx_loss_min_idx = list()
##mouse_idx_min_loss = list()
##num_test_mice = split_idx.shape[1]
#
#ID_idx = list()
#ID_idx_loss = list()
#ID_idx_loss_min_idx = list()
#best_mdl_idx = list()
#num_test_mice = split_idx.shape[1]
#Yte_best = np.zeros((D.shape[0], D.shape[1]))
#rest_mdl_idx = list()
#
#for i in range(0,D.shape[0]):       #Loop through all 68 mice IDs (1...68)
#    #Extract the rows/cols in test_idx to each mouse ID (1...68)
#    ID_idx.append(np.array(np.where(test_idx == i)))    #the i:th item in the list of nd arrays (ID_idx) will contain the coordinates (rows and cols) in test_idx to the i:th mouse ID. 
#    
#    #For each ID, find the iteration with the minimum loss from loss_vl_min (equal loss for all columns in test_idx)
#    ID_idx_loss.append(np.array(loss_vl_min[ID_idx[i][1]]))                   #For each mouse (row = i), store the column [1] for which iteration it was in the test set. Each column is associated with the same loss. Ex. if mouse_idx[0] outputs array([21,  2, 18]), this means that mouse_idx[0] was in the test set during iteration 21, 2 and 18, and is associated with loss corresponding to these  indices in the minimum of the loss vector)]
#    ID_idx_loss_min_idx.append((np.array(np.argmin(ID_idx_loss[i]))))         #This is the index of the minimum value, refering to each row in the mouse_idx array    
#    
#    #Find best model index
#    best_mdl_idx.append(np.array(ID_idx[i][:,ID_idx_loss_min_idx[i]]))          #This can perhaps be done more beatiful outside the for loop. Anyways, this picks out the coordinates (rows/cols) for each mouse (row) with the lowest loss 
#    
#    #Find the minimum number of mice 
#    if ID_idx[i].shape[1] < num_test_mice:                               #This if loop can also be implemented better, but it picks out the limiting number of mice in the test set
#        num_test_mice = ID_idx[i].shape[1] 
#
#    #Pick out best model for each mouse
#    Yte_best[i] = Yte_it[best_mdl_idx[i][0]][best_mdl_idx[i][1]]
#    
#    #Remove the best model from ID_idx and put the rest in rest_mdl_idx
#    rest_mdl_idx.append(np.array(np.delete(ID_idx[i],ID_idx_loss_min_idx[i],1)))   #Delete the column given in ID_idx_loss_min_idx[i] (syntax for delete is delete(array, which object, along which axis)) See: https://stackoverflow.com/questions/1642730/how-to-delete-columns-in-numpy-array
#
##Random sample num_test_mice-1 number of samples from the rest_mdl_idx for each mouse
##num_test_mice = 3       #Override just for testing. DELETE THIS LINE LATER!!
#Yte_rest = np.zeros((D.shape[0], D.shape[1], num_test_mice-1))
#idx_rest = list()
#
#for i in range(0,D.shape[0]):
#    
#    #Generate random number of columns
#    rnd_idx = random.sample(range(0, rest_mdl_idx[i].shape[1]), num_test_mice-1)    #Ex. pick 2 random samples between 0 and 4: random.sample(range(0, 4), 2)
#    
#    #Pick out the random columns
#    idx_rest.append(np.array(rest_mdl_idx[i][:,rnd_idx]))   #Replace 5 by i later.
#    
#    #Assign YTE_rest to the curves with indexes in idx_rest.
#    Yte_rest[i] = Yte_it[idx_rest[i][0], idx_rest[i][1]].T
#    
##Concatenate Yte_best and Yte_rest and create best_mdl_idx
#Yte_best = np.expand_dims(Yte_best, axis=2)             #Add a singelton dimension
#results = np.concatenate((Yte_best, Yte_rest), axis=2)  #Concatenate
#results = np.transpose(results, (2,0,1))                #Permute to match Kristoffers work for easy import to Matlab
#best_model_idx = np.zeros(D.shape[0])                    #Best model is always the first element in this implementation
#
##%% Save the variables
#sio.savemat('/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/LSTM_results_Keras_l1_with_scaling_Fang2008_weights.mat', {'results':results})
#sio.savemat('/Volumes/sku014/PhD/Projekt/MLDIF-Kristoffer/best_LSTM_model_idx_Keras_l1_with_scaling_Fang2008_weights.mat', {'best_model_idx':best_model_idx})

#%% Rest code   
#    #Return the IDs of 0..68 mice! - not needed, they are in the correct order of VOIid
#    
#    
##    Yte_it[9][21]
##    test = Yte_it[[9, 2],[21,16]]
#     
##plt.plot(time_scale, yte[0,:])
##plt.plot(time_scale, Yte_best[0,:])   
##plt.plot(time_scale, Yte_rest[0,:,:]) 
##plt.legend(['GT', 'Best', 'Rest', 'Rest'], loc='upper right')  
##
##plt.plot(yte[0,:])
##plt.plot(Yte_best[0,:])   
##plt.plot(Yte_rest[0,:,:]) 
##plt.legend(['GT', 'Best', 'Rest', 'Rest'], loc='upper right')  
#    
#    [best_mdl_idx[i][1]]
#    #rnd_idx = np.random.permutation()  #If 10 mice is limiting, one is the best model, pull out 9 of the rest randomly
#    
#    A = np.array(([1,2,3],[4,5,6],[7,8,9]))
#    A_idx = np.array([0,2]) 
#    A[:,A_idx]
#    
#    
#    
#    #mouse_idx_it.append(np.where(test_idx == i)[1])                        #For each mouse (row = i), store the column [1] for which iteration it was in the test set. Each column is associated with the same loss. Ex. if mouse_idx[0] outputs array([21,  2, 18]), this means that mouse_idx[0] was in the test set during iteration 21, 2 and 18, and is associated with loss corresponding to these  indices in the minimum of the loss vector)]
#    #mouse_idx_row.append(np.where(test_idx == i)[0])
#    #mouse_idx_loss.append(loss_vl_min[mouse_idx_it[i]])
#    #mouse_idx_loss_min_idx.append(np.argmin(mouse_idx_loss[i]))         #This is the index of the minimum value, refering to each row in the mouse_idx array 
#    #mouse_idx_min_loss.append(mouse_idx[i][mouse_idx_loss_min_idx[i]])  #This can perhaps be done more beatiful outside the for loop. Anyways, this picks out the iteration number for each mouse (row) with the lowest loss
#                              
#    
##CONTINUE:
##Create an Yte_it array but with num_test_mice number in third dimension with the best model first, then randomly selected models as rest  
#        #NEED TO EXTRACT BEST MODEL ROW NUMBER AS WELL
#        #GO BACK TO MODEL WITH 30 MICE!
#Yte_out = np.zeros((Yte_it.shape[0], _it, D.shape[1], split_idx.shape[1]))
#[mouse_idx_row[i], mouse_idx_it[i]]
##For each mouse, return an array of indices, corresponding to the model with the lowest loss
##Done in for loop above
#
##Then pick out num_test_mice number of random models
#        
#    
##Get the minimum number of results for each row (limiting)    
##num_test_mice = np.min(map(len, mouse_idx))
#
#mouse_idx = np.asarray(mouse_idx)       #Convert to numpy array to handle a bit better
#len(mouse_idx[:])
#
#len(mouse_idx[:])
#
#loss_val_min[mouse_idx[:]]
#
#np.min(loss_vl_min[mouse_idx])     #loss
#np.argmin(loss_vl_min[mouse_idx])  #index

#mouse_idx = np.array(mouse_idx)

#%% Plot all predictions
#plt.figure(figsize=(10,6))
#for i in range(0,len(yte)):
#    plt.subplot(2, 1, 1)
#    plt.plot(Yte[i,:])
#    plt.title('All test mice')
#    plt.legend(['LV+VC'], loc='upper right')
#       
#    plt.subplot(2, 1, 2)
#    plt.plot(yte[i,:])
#    plt.title('All test mice')
#    plt.legend(['prediction'], loc='upper right')
#
#plt.tight_layout()                  #Use tight layout
#plt.show()

#%% plot baseline and predictions for a random mouse
#plt.plot(scaler.inverse_transform(dataset))
#mouse_tr = random.randint(0,len(Ytr)-1)
#mouse_te = random.randint(0,len(yte)-1)
#plt.figure(figsize=(10,6))
############## Uncomment to plot with UNIT time scale #############
#plt.subplot(2, 1, 1)
#plt.plot(Ytr[mouse_tr,:])
#plt.plot(ytr[mouse_tr,:])

#plt.subplot(2, 1, 2)
#plt.plot(Yte[mouse_te,:])
#plt.plot(yte[mouse_te,:])

############# Uncomment to plot with TRUE time scale #############
#plt.subplot(2, 2, 1)
#plt.plot(time_scale, Ytr[mouse_tr,:])
#plt.plot(time_scale, ytr[mouse_tr,:])

#plt.title('0-60 min. Training Mouse ' + str(IDtr[mouse_tr]))
#plt.legend(['LV+VC', 'prediction'], loc='upper right')

#plt.subplot(2, 2, 2)
#plt.plot(time_scale[0:33], Ytr[mouse_tr,0:33])
#plt.plot(time_scale[0:33], ytr[mouse_tr,0:33])

#plt.title('0-5 min. Training Mouse ' + str(IDtr[mouse_tr]))
#plt.legend(['LV+VC', 'prediction'], loc='upper right')

#plt.subplot(1, 2, 1)
#plt.plot(time_scale, Yte_it[mouse_te,999,:])
#plt.plot(time_scale, yte[mouse_te,:])
#
#plt.title('0-60 min. Test Mouse ' + str(IDte[mouse_te]))
#plt.legend(['LV+VC', 'prediction'], loc='upper right')
#
#plt.subplot(1, 2, 2)
#plt.plot(time_scale[0:33], Yte_it[mouse_te,:,0:33])
#plt.plot(time_scale[0:33], yte[mouse_te,0:33])
#
#plt.title('0-5 min. Test Mouse ' + str(IDte[mouse_te]))
#plt.legend(['LV+VC', 'prediction'], loc='upper right')
#
#plt.tight_layout()                  #Use tight layout
#plt.show()

    
#%%

#plt.plot(loss_tr[:,1])
#plt.plot(loss_vl[:,1])
#plt.title('Model Loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.show()