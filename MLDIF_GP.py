######################################################################################################################################################################################################
# Machine Learning-Derived Input-Function in Dynamic 18F-FDG PET 
#
# MLDIF_GP.py
#
# This code performs Gaussian Processes (GP) model training on dynamic FDG PET data for arterial input function (AIF) regression.
# The model takes time-activity-curves from an arbitrary number of tissue regions as input, trains a model that predicts the AIF, required for further compartment modeling.
# A ground truth AIF is required for model training, which can be obtained from blood or image data.
# Two GP models are evailated, GP1 and GP2.
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

import tensorflow as tf
from MLDIF_functions import load_data
from MLDIF_functions import normalize_data
import gpflow
import numpy as np
import os

#%% Define load and save_path
path = os.getcwd()
load_path = path + '/data.mat'
save_path = path + '/GP_AIF_regression/'

#%% Load the data

#The following loads the data set used in the paper, but you may equally well define your own data loading function. 
D, Y, VOIid, time_scale = load_data(load_path,2)    #Use second ground truth (Fang2008-weights)
D, scaler = normalize_data(D)                       #Data is normalized for GP!

#%% Start leave one out cross validation

#Initialize predictions
Y_mean_gp1 = np.zeros((D.shape[0], D.shape[1]))
Y_var_gp1  = np.zeros((D.shape[0], D.shape[1]))

Y_mean_gp2 = np.zeros((D.shape[0], D.shape[1]))
Y_var_gp2  = np.zeros((D.shape[0], D.shape[1]))

for te in range(0,D.shape[0]):
    
    print(te)
    
    #Extract current test mouse idx and concatenate rest into training set indices
    tr1 = np.arange(te)
    tr2 = np.arange(te+1,D.shape[0])
    tr = np.concatenate((tr1,tr2))
    
    Xtr, Xte = D[tr], D[te]
    ytr, yte = Y[tr], Y[te] 
    
    #%% GP 1: Assume independence and reshape dataset into (67x44,org) independent (idp) representation.
    tf.Session.reset        #Reset tensorflow session
    
    Xtr_gp1 = np.reshape(Xtr,(Xtr.shape[0]*Xtr.shape[1], Xtr.shape[2])).astype(float)         
  
    Ytr_gp1 = np.reshape(ytr,(ytr.shape[0]*ytr.shape[1], 1)).astype(float)         

    #Set up Gaussian Process model
    
    #Define kernel model and optimizer
    k_gp1 = gpflow.kernels.Matern52(input_dim=10, ARD=True)   #Set ARD=True to be able to set lengthscale for each feature dimension.
    m_gp1 = gpflow.models.GPR(Xtr_gp1, Ytr_gp1, kern=k_gp1)
    opt_gp1 = gpflow.train.ScipyOptimizer()
    opt_gp1.minimize(m_gp1)
    
#    m_gp1.as_pandas_table()     #Uncomment to show info about the model
    
    #Make predictions of the test sample
    Y_mean, Y_var = m_gp1.predict_y(Xte)
    
    #Put together results
    Y_mean_gp1[te,:] = np.squeeze(Y_mean)
    Y_var_gp1[te,:] = np.squeeze(Y_var)
    
    #%% GP 2: Output the whole time vector at once!
    tf.Session.reset        #Reset tensorflow session

    #Flatten the data
    Xtr_gp2 = np.reshape(Xtr,(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])).astype(float)
    Xte_gp2 = np.reshape(Xte,(1, Xte.shape[0]*Xte.shape[1])).astype(float)
    
    #Define kernel model and optimizer  
    k_gp2 = gpflow.kernels.Matern52(input_dim=440, ARD=True)
    m_gp2 = gpflow.models.GPR(Xtr_gp2, ytr, kern=k_gp2)
    opt_gp2 = gpflow.train.ScipyOptimizer()
    opt_gp2.minimize(m_gp2)
    
#    m_gp2.as_pandas_table()     #Uncomment to show info about the model
    
    #Make predictions of the test sample
    Y_mean, Y_var = m_gp2.predict_y(Xte_gp2)
    
    Y_mean_gp2[te,:] = np.squeeze(Y_mean)
    Y_var_gp2[te,:]  = np.squeeze(Y_var)
    
#%%End of GP training and prediction for all mice
# mean and var is now in Y:mean_gp1(2) and Y_var_gp1(2). 
#Concatenate the results and save:
Y_mean_gp1_exp = np.expand_dims(Y_mean_gp1, axis=0)             #Add a singelton dimension
Y_var_gp1_exp = np.expand_dims(Y_var_gp1, axis=0)    
Y_mean_gp2_exp = np.expand_dims(Y_mean_gp2, axis=0)    
Y_var_gp2_exp = np.expand_dims(Y_var_gp2, axis=0)    

Y_GP = np.concatenate((Y_mean_gp1_exp, Y_var_gp1_exp, Y_mean_gp2_exp, Y_var_gp2_exp), axis=0)  #Concatenate
sio.savemat(save_path + '/GP_results.mat', {'Y_GP':Y_GP})