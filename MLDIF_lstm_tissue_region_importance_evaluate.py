######################################################################################################################################################################################################
# Machine Learning-Derived Input-Function in Dynamic 18F-FDG PET 
#
# MLDIF_lstm_tissue_region_importance_evaluate.py
#
# This code performs model evaluation of previously trained LSTM models from dynamic FDG PET data for arterial input function (AIF) prediction, using all different combinations of input tissue regions (features).
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
import scipy.io as sio
from MLDIF_functions import split
from keras.models import load_model
from keras import backend as K
import pickle
import itertools
import os

#%% Load variables
path = os.getcwd()
save_path = path + '/LSTM_tissue_region_importance/'

with open(save_path + 'variables.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loss_tr, loss_vl, D, Y, VOIid, time_scale, split_idx, exp_mat = pickle.load(f)

#%% Load saved models one by one and make +store the predictions
Yte_exp = np.zeros((12, D.shape[1], len(exp_mat)))

k = 1                           #k is the number of times the tissue combination experiment has been performed.
t = np.arange(0,k,1)            #This is all features to be combined
experiment = 0                  #Experiment counter. To become 1023

#Loop all number of features (numfeat): 1, 2, 3...10
for numfeat in range(1,11):
    print(numfeat)

    c = list(itertools.combinations(t, numfeat))     #c is a list of all possible combinations of numfeat elements from t.
    
    
    #Loop all list elements of c
    for el in range(0,len(c)):

        print(experiment)
               
        D_extr = D[:,:,c[el]]       #D_extracted contains selected features only, as given in the list c. 
    
        #Clear the last session from memory
        K.clear_session()
        
        #Load the model
        model = load_model(save_path + '/MLDIF_lstm_tissue_region_importance_exp_' + str(experiment) + '.h5')
        
        #Obtain the current training and test set
        Xtr, Xvl, Xte, ytr, yvl, yte, IDtr, IDvl, IDte  = split(D_extr,Y,VOIid, split_idx[:,0])
        
        #Make predictions and invert them
        Yte = model.predict(Xte)
        
        Yte = Yte.squeeze()
               
        #Store the predictions from the current experiment
        Yte_exp[:,:,experiment] = Yte  

        experiment += 1

#%% Calculate the RMSE for all models

RMSE_mat = np.sqrt(((Yte_exp - yte) ** 2).mean(axis=1))

mean_RMSE = RMSE_mat.mean(axis=0)   #RMSE over all test mice in this particular shuffling

#%% Save variables

with open(save_path + '/predictions.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Yte_exp, exp_mat, RMSE_mat], f)

#%% Save the results to a Matlab file for further post processing
sio.savemat(save_path + '/results/RMSE_mat.mat', {'RMSE_mat':RMSE_mat})
sio.savemat(save_path + '/results/exp_mat.mat', {'exp_mat':exp_mat})
sio.savemat(save_path + '/results/Yte_exp.mat', {'Yte_exp':Yte_exp})
sio.savemat(save_path + '/results/IDte.mat', {'IDte':IDte})
sio.savemat(save_path + '/results/split_idx.mat', {'split_idx':split_idx})