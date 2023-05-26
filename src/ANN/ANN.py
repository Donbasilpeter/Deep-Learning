# import neeeded libraries for the ANN
import numpy as np
import pandas as pd
import tensorflow as tf

# preprocessing the given data for training purposes
dataset = pd.read_csv('./Data/Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values

