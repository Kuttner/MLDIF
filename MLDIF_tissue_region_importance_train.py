######################################################################################################################################################################################################
# Machine Learning-Derived Input-Function in Dynamic 18F-FDG PET 
#
# MLDIF_tissue_region_importance_train.py
#
# This code performs LSTM model training on dynamic FDG PET data for arterial input function (AIF) prediction using all different combinations of input tissue regions (features).
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
from MLDIF_functions import lstm_train_fkn
from MLDIF_functions import load_data
from MLDIF_functions import split
from keras import backend as K
import os
import pickle
import itertools

#%% Define load and save_path
path = os.getcwd()
load_path = path + '/data.mat'
save_path = path + '/LSTM_tissue_region_importance/'


#%% Load the data

#The following loads the data set used in the paper, but you may equally well define your own data loading function. 
D, Y, VOIid, time_scale = load_data(load_path,2)

#%% Define some hyperparameters for the LSTM model
it = 1              #Iterations, 1 (Test all combinations for one particular shuffle/split)
n_epochs = 1000     #Maximum number of epochs, 1000
min_delta = 0.0001  #The minimum change in the validation loss to qualify as an improvement, 0.0001
patience = 50       #The number of epochs with no improvement after which training will be stopped, 50
batch_size = 12     #Mini-batch size, 12

#%% Shuffle the data set and create split index array
split_idx = np.ndarray((68,it),int)
for i in range(0,it):
    split_idx[:,i] = np.random.permutation(68)

#%% Start LSTM training with different tissue region combinations

loss_tr = list()
loss_vl = list()

t = np.arange(0,10,1)           #This is all features to be combined
experiment = 0                  #Experiment counter. To become 1023.
exp_mat = list()                #Initialize the experiment matrix

#Loop all number of features (numfeat): 1, 2, 3...10
for numfeat in range(1,11):         
    print(numfeat)
    
    c = list(itertools.combinations(t, numfeat))     #c is a list of all possible combinations of numfeat elements from t.
    
    #Loop all list elements of c
    for el in range(0,len(c)):
        print(el)
        
        exp_mat.append([[numfeat], c[el]])                                                          # Store the number of features and current combination in the experiment matrix.
        
        D_extr = D[:,:,c[el]]                                                                       #D_extracted contains selected features only, as given in the list c.
        Xtr, Xvl, Xte, Ytr, Yvl, Yte, IDtr, IDvl, IDte  = split(D_extr,Y,VOIid, split_idx[:,i])     #Extract training, validation and test data with only the current features, given in c[el]

        #Clear the last session from memory
        K.clear_session()
        
        # Train the LSTM model
        model_save_path = save_path + '/MLDIF_lstm_tissue_region_importance_exp_' + str(experiment) + '.h5'
        history = lstm_train_fkn(Xtr, Ytr, Xvl, Yvl, min_delta, patience, n_epochs, batch_size, model_save_path)
        
        #Save the training and validation loss for each iteration.
        loss_tr.append(history.history['loss'])
        loss_vl.append(history.history['val_loss'])

        experiment += 1

#%% Save other variables 

with open(save_path + '/variables.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([loss_tr, loss_vl, D, Y, VOIid, time_scale, split_idx, exp_mat], f)
