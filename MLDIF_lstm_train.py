######################################################################################################################################################################################################
# Machine Learning-Derived Input-Function in Dynamic 18F-FDG PET 
#
# This code performs LSTM model training on dynamic FDG PET data for arterial input function (AIF) prediction.
# The model takes time-activity-curves from an arbitrary number of tissue regions as input, trains a model that predicts the AIF, required for further compartment modeling.
# A ground truth AIF is required for model training, which can be obtained from blood or image data.
# Please refer to the publication by Kuttner et al 2019 for further details.
# 
# Input:    D, the input data, containing the time-actiity-curves for all samples and time steps. An ndarray with size (batch_size, timesteps, input_dim)
#           Y, the output data, containing the ground truth AIF for all samples and time steps. A float with size (batch_size, timesteps, 1)
#           VOIid, a vector with 
#           time_scale, a vector with the time stamps for each time step. A float with size (timesteps,). The models assume equal time step intervals between samples, but not necessarily uniform within each sample. This variable is only required for plotting/visualization purposes. 
#
# Output:   MLDIF_lstm_it_x.h5, a Keras model file for each iteration, x, saved in the folder defined by save_path.
#           variables.pkl, a variables bundle containing the necessary files for MLDIF_lstm_evaulate.py   
#
# Samuel Kuttner, samuel.kuttner@uit.no
#
######################################################################################################################################################################################################    

import numpy as np
import matplotlib.pyplot as plt
from MLDIF_lstm_functions import lstm_train_fkn
from MLDIF_lstm_functions import load_data
from MLDIF_lstm_functions import normalize_data
from MLDIF_lstm_functions import split
from keras.models import load_model
from keras import backend as K
import os, datetime
import random
import pickle


#%% Define load and save_path
load_path = 'data.mat'
save_path = '/Users/sku014/Documents/Temp/test'

#%% Load the data

#The following loads the data set used in the paper, but you may equally well define your own data loading function. 
D, Y, VOIid, time_scale = load_data(load_path,2)

#%% Define some hyperparameters for the LSTM model
it = 1000           #Iterations, 1000
n_epochs = 1000     #Maximum number of epochs, 1000
min_delta = 0.0001  #The minimum change in the validation loss to qualify as an improvement, 0.0001
patience = 50       #The number of epochs with no improvement after which training will be stopped, 50
batch_size = 12     #Mini-batch size, 12


#%% Shuffle the data set and create split index array
split_idx = np.ndarray((68,it),int)
for i in range(0,it):
    split_idx[:,i] = np.random.permutation(68)
    

#%% Start all 1000 iterations

loss_tr = list()
loss_vl = list()

for i in range(0,it):
    print(i)
    #Extract current training, validation and test data set
    Xtr, Xvl, Xte, Ytr, Yvl, Yte, IDtr, IDvl, IDte  = split(D,Y,VOIid, split_idx[:,i])
    
    #Clear the last session from memory
    K.clear_session()
    
    # Train the LSTM model
    model_save_path = save_path + '/MLDIF_lstm_it_' + str(i) + '.h5'
    history = lstm_train_fkn(Xtr, Ytr, Xvl, Yvl, min_delta, patience, n_epochs, batch_size, model_save_path)
    
    #Save the training and validation loss for each iteration.
    loss_tr.append(history.history['loss'])
    loss_vl.append(history.history['val_loss'])

#%% Save other variables 

with open(save_path + '/variables.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([loss_tr, loss_vl, D, Y, VOIid, time_scale, split_idx], f)   
