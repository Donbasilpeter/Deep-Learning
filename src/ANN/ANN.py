# import neeeded libraries for the ANN
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ANN:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_train = None
        self.X = None
        self.Y = None

    
    def preprocess(self):
        self.__load_data()
        self.__label_encode()
        self.__split_dataset()
        self.__feature_scaling()
        

    def  __load_data(self):
        self.dataset = pd.read_csv(self.data_path)
        self.X = self.dataset.iloc[:, 3:-1].values
        self.Y = self.dataset.iloc[:, -1].values

    def __label_encode(self):
        le = LabelEncoder()
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        self.X[:, 2] = le.fit_transform(self.X[:, 2])
        self.X = np.array(ct.fit_transform(self.X))
    
    def __split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_train = train_test_split(self.X, self.Y, test_size = 0.2, random_state = 0)
        
    def __feature_scaling(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        print(self.X_train)
        
        

