######################################################################################################################################################################################################
# Machine Learning-Derived Input-Function in Dynamic 18F-FDG PET 
#
# This code performs model evaluation of previously trained LSTM models from dynamic FDG PET data for arterial input function (AIF) prediction.
# The script evaluates the trained model on the test data and returns a predicted AIF for each model.
# The script returns the AIF from best model (lowest validation loss) in the first index, as well as an equal number of AIFs for all samples, randomly choosen from the remaining model predictions.  
# Please refer to the publication by Kuttner et al 2019 for further details.
# 
# Input:    variables.pkl, contining all necessary variables saved in MLDIF_lstm_train.py after model trianing.
#
# Output:   results.mat, a MATLAB data file containing the best LSTM model prediction (with lowest validation loss) in the first index, followed by the rest of the AIF predicitons
#
# Samuel Kuttner, samuel.kuttner@uit.no
#
######################################################################################################################################################################################################    

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from lstm_functions import split
from keras.models import load_model
from keras import backend as K
import random
import pickle
import os


#%% Load variables
path = os.getcwd()
save_path = path + '/aif_prediction/'

with open(save_path + 'variables.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loss_tr, loss_vl, D, Y, VOIid, time_scale, split_idx = pickle.load(f)

#%% Load saved models one by one and make + store the predictions
Yte_it = np.zeros((12, D.shape[1], split_idx.shape[1]))

for i in range(0,split_idx.shape[1]):

    print(i)    
    
    #Clear the last session from memory
    K.clear_session()
    
    #Load the model
    model = load_model(save_path + '/1-layer-lstm_it_' + str(i) + '.h5')
    
    #Obtain the current training and test set
    Xtr, Xvl, Xte, ytr, yvl, yte, IDtr, IDvl, IDte  = split(D,Y,VOIid, split_idx[:,i])      #Not sure that IDtr, IDvl and IDte are needed. Check when everything is done. All is handled through split_idx and the split function
    
    #Make predictions and invert them
    Yte = model.predict(Xte)
    
    Yte = Yte.squeeze()
        
    #Store the predictions from the i:th iteration
    Yte_it[:,:,i] = Yte  


    
#%% Save variables to disk.

# Saving the objects:
with open(save_path + '/predictions.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Yte_it], f)
    
#%% Load Yte_it in case work is resumed from here.
    
with open(save_path + 'predictions.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    Yte_it = np.array(pickle.load(f))
Yte_it = np.squeeze(Yte_it)

#%%  Process the results to extract the best model and equal number of remaining models for output.

loss_vl_min = np.zeros(1000)

for i in range(0,1000):                         #New method working with loss_vl as list.
    loss_vl_min[i] = min(loss_vl[i])
    
# Extract the indexes of the test set
test_idx = split_idx[-12:,:]

#Reshape Yte_it to the same shape az test_idx
Yte_it = np.transpose(Yte_it, (0,2,1))  #E.g. have the shape: (12,100,44)

ID_idx = list()
ID_idx_loss = list()
ID_idx_loss_min_idx = list()
best_mdl_idx = list()
num_test_mice = split_idx.shape[1]
num_test_mice_max = 0
Yte_best = np.zeros((D.shape[0], D.shape[1]))
rest_mdl_idx = list()

for i in range(0,D.shape[0]):       #Loop through all 68 mice IDs (1...68)
    
    #Extract the rows/cols in test_idx to each mouse ID (1...68)
    ID_idx.append(np.array(np.where(test_idx == i)))    #the i:th item in the list of nd arrays (ID_idx) will contain the coordinates (rows and cols) in test_idx to the i:th mouse ID. 
    
    #For each ID, find the iteration with the minimum loss from loss_vl_min (equal loss for all columns in test_idx)
    ID_idx_loss.append(np.array(loss_vl_min[ID_idx[i][1]]))                   #For each mouse (row = i), store the column [1] for which iteration it was in the test set. Each column is associated with the same loss. Ex. if mouse_idx[0] outputs array([21,  2, 18]), this means that mouse_idx[0] was in the test set during iteration 21, 2 and 18, and is associated with loss corresponding to these  indices in the minimum of the loss vector)]
    ID_idx_loss_min_idx.append((np.array(np.argmin(ID_idx_loss[i]))))         #This is the index of the minimum value, refering to each row in the mouse_idx array    
    
    #Find best model index
    best_mdl_idx.append(np.array(ID_idx[i][:,ID_idx_loss_min_idx[i]]))          #Picks out the coordinates (rows/cols) for each mouse (row) with the lowest loss 
    
    #Find the minimum number of mice 
    if ID_idx[i].shape[1] < num_test_mice:                                      #Pick out the limiting number of mice in the test set
        num_test_mice = ID_idx[i].shape[1] 
    
    #Find the maximum number of mice 
    if ID_idx[i].shape[1] > num_test_mice_max:                                  #Picks out the limiting number of mice in the test set
        num_test_mice_max = ID_idx[i].shape[1] 
        
    #Pick out best model for each mouse
    Yte_best[i] = Yte_it[best_mdl_idx[i][0]][best_mdl_idx[i][1]]
    
    #Remove the best model from ID_idx and put the rest in rest_mdl_idx
    rest_mdl_idx.append(np.array(np.delete(ID_idx[i],ID_idx_loss_min_idx[i],1)))   #Delete the column given in ID_idx_loss_min_idx[i]

#Random sample num_test_mice-1 number of samples from the rest_mdl_idx for each mouse
Yte_rest = np.zeros((D.shape[0], D.shape[1], num_test_mice-1))
idx_rest = list()

for i in range(0,D.shape[0]):
    
    #Generate random number of columns
    rnd_idx = random.sample(range(0, rest_mdl_idx[i].shape[1]), num_test_mice-1)    #Ex. pick 2 random samples between 0 and 4: random.sample(range(0, 4), 2)
    
    #Pick out the random columns
    idx_rest.append(np.array(rest_mdl_idx[i][:,rnd_idx]))   #Replace 5 by i later.
    
    #Assign YTE_rest to the curves with indexes in idx_rest.
    Yte_rest[i] = Yte_it[idx_rest[i][0], idx_rest[i][1]].T
    
#Concatenate Yte_best and Yte_rest and create best_mdl_idx
Yte_best = np.expand_dims(Yte_best, axis=2)             #Add a singelton dimension
results = np.concatenate((Yte_best, Yte_rest), axis=2)  #Concatenate
results = np.transpose(results, (2,0,1))                #Permute to be compatible with earlier work.
best_model_idx = np.zeros(D.shape[0])                    #Best model is always the first element in this implementation

#%% Save the variables
sio.savemat('results.mat', {'results':results})
sio.savemat('best_model_idx.mat', {'best_model_idx':best_model_idx}) #This file is needed only for compability with the Matlab post processing script.
