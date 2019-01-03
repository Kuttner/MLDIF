######################################################################################################################################################################################################
# Machine Learning-Derived Input-Function in Dynamic 18F-FDG PET 
#
# MLDIF_functions.py
#
# This code bundle contains functions required for LSTM model training and evaulation.
# Please refer to the main script files and the publication by Kuttner et al 2019 for further details.
# 
# Samuel Kuttner, samuel.kuttner@uit.no
#
######################################################################################################################################################################################################  


#%%############################### LOAD DATA ###############################
# Note that the data set for this paper was prepared and preprocessed in Matlab, and subsequently loaded into Python using the function below. 
# Other data sets may be implemented using other data loading functions.
   
import numpy as np
import scipy.io as sio


def load_data(path, case):

    data = sio.loadmat(path)
       
    # Extract the variables to be used
    D_raw = data['D']
    VOIid = data['VOIid']
    Y_raw = data['Y']
    
    #############  Use VC for ground truth #############
    if case == 0:
        Y = Y_raw[0,0]                                        # Extract the first  IF as ground truth first (VC)
        D = D_raw[:,:,1:12,:]                                 # Remove the VC from D, since it is used as ground truth 
    
    ############# Use min(LV,VC) for ground truth ############# 
    if case == 1:
        Y = Y_raw[0,1]                                         # Extract the second IF as ground truth first min(LV,VC)
        D = np.delete(D_raw, [0,2],axis=2)                     # Remove VC (0) and LV (2) from D, since it is used as ground truth 
    
    ############# Use Fang2008 model fit for ground truth #############
    if case == 2:
        Y = Y_raw[0,3]                                         # Extract the third IF as ground truth first Fang2008 Eq8 model fit to min(LV,VC)
        D = np.delete(D_raw, [0,2],axis=2)                     # Remove VC (0) and LV (2) from D, since it is used as ground truth 
    
    #####################################################################################
    time_scale = Y[:,0,0]                                  # Extract the time scale
    Y = Y[:,1,:]                                           # Extract just the data points from the Y vector (omit the time points for now...) 
    D = D[:,4,:,:]                                         # Also extract only the 4th column (average) from D, since this will be used for training
    D = np.transpose(D, (2, 0, 1))                         # Reshape D and Y to the format [batch_size, timesteps, input_dim]. This works equally well: #D = np.reshape(D, (D.shape[2], D.shape[0], D.shape[1]))      # Reshape D to the format [batch_size, timesteps, input_dim].
    Y = Y.T                                                #Transpose Y. 
    Y = np.reshape(Y,(Y.shape[0], Y.shape[1],1))           #Form the Y variable to have the same size as D.
    
    return D, Y, VOIid, time_scale
    

#%%############################### NORMALIZE DATA ###############################
from sklearn.preprocessing import MinMaxScaler
    
def normalize_data(D):
    scaler = MinMaxScaler(feature_range=(0,1))
    
    for i in range(0, len(D)):
        D[i,:,:] = scaler.fit_transform(D[i,:,:])
    
    return D,scaler
        
#%%############################### SPLIT INTO TRAINING/VALIDATION/TEST SET (44/12/12) ###############################
    
def split(D,Y, VOIid, idx):    
    
    tr = idx[0:44]
    vl = idx[44:56]
    te = idx[56:len(idx)]
    
    Xtr, Xvl, Xte = D[tr], D[vl], D[te] 
    Ytr, Yvl, Yte = Y[tr], Y[vl], Y[te] 
    IDtr, IDvl, IDte = VOIid[tr], VOIid[vl], VOIid[te] 
    
    return Xtr, Xvl, Xte, Ytr, Yvl, Yte, IDtr, IDvl, IDte
   
#%%############################### LSTM MODEL ###############################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def lstm_train_fkn(Xtr, Ytr, Xvl, Yvl, min_delta, patience, n_epochs, batch_size, model_save_path):
    model = Sequential()
    model.add(LSTM(20, input_shape=(44,Xtr.shape[2]), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Define early stopping callback
    earlystop = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto')
    mcp = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, mode="min", save_weights_only=False)                #Return the best possible model, from: https://github.com/keras-team/keras/issues/2768
    callbacks_list = [earlystop, mcp]

    
    history = model.fit(Xtr, Ytr, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=0, validation_data=(Xvl, Yvl), shuffle=True)
    
    return history      