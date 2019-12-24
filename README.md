# Machine Learning-Derived Input-Function (MLDIF) in Dynamic <sup>18</sup>F-FDG PET
This repository contains source code for non-invasive prediction of the arterial input function (AIF) in small-animal dynamic FDG PET studies.

Two well-known, machine learning-based regression models, have been implemented, based on Gaussian processes (GP) and long short-term memory (LSTM) recurrent neural network, respectively.

## Usage
The followings files are available:
- MLDIF lstm_train.py:     Train an LSTM model with training and validation data. 
- MLDIF_lstm_evaluate.py:  Evaluate a trained LSTM model on test data.           
- MLDIF_tissue_region_importance_train.py:    Train an LSTM model with different feature combinations.
- MLDIF_tissue_region_importance_evaluate.py:  Evaluate an LSTM model trained with feature combinations.
- MLDIF_GP.py: Train and evaluate two different GP models

Please see the comments in the code, and the puplication below, for further implementation details.

## Citation
Please cite any usage of the content of this repository as:

Kuttner S, Knutsen Wickstrøm K, Kalda G, Dorraji SE, Martin-Armas M, Oteiza A, et al. Machine learning derived input-function in a dynamic 18F-FDG PET study of mice. Biomed Phys Eng Express. 2019 Dec 20;1–46. Available from: https://doi.org/10.1088/2057-1976/ab6496
