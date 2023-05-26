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
    def __init__(self, data_path,model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X = None
        self.Y = None
        self.output =None 
        
        try:    
            self.preprocess()
            self.ann = tf.keras.models.load_model(model_path)
            print("ANN model is loaded...")
            while(1):  
                user_input  =  input("should I retrain the model? Y/N ") 
                if user_input =="Y" :
                    self.ann = None
                    self.neural_network()
                    break
                    
                elif user_input =="N":
                    break
                
                else :
                    print("Please give a valid response...!")
                
                
        except:
            print("No ANN model is found. Creating one...!")
            self.ann = None
            self.neural_network()
            
            

    def preprocess(self):
        """ Preprocesses the data
        Here, we call some private subfunctions 
        to process our input data to 
        make it more pricese
        """
        self.__load_data()
        self.__label_encode()
        self.__split_dataset()
        self.__feature_scaling()

    def __load_data(self):
        """Loads th data from csv to the class
        """
        self.dataset = pd.read_csv(self.data_path)
        self.X = self.dataset.iloc[:, 3:-1].values
        self.Y = self.dataset.iloc[:, -1].values

    def __label_encode(self):
        """Enocdes the gender coloum and location coloum with numernical values for 
        better pricision
        """
        le = LabelEncoder()
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        self.X[:, 2] = le.fit_transform(self.X[:, 2])
        self.X = np.array(ct.fit_transform(self.X))

    def __split_dataset(self):
        """split the dataset for training and testing purposes
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=0)

    def __feature_scaling(self):
        """feature scaling the training data.
        """
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def neural_network(self):
        """Create and train the neural network for the specific set of data
        """
        self.__create_neural_network()
        self.__compile_neural_network()
        self.__train_neural_network()

    def __create_neural_network(self):
        self.ann = tf.keras.models.Sequential() # initialise the neural network
        self.ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  #add first layer of neural network
        self.ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # add another hidden layer
        self.ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #final layer with one neuron with sigmoid activation function
        
    def __compile_neural_network(self):
        """compile the created neural network 
        """
        self.ann.compile(optimizer="adam", loss ='binary_crossentropy', metrics = ["accuracy"] )
        
    def __train_neural_network(self):
        """train and save the model using the specified data
        """
        self.ann.fit(self.X_train, self.Y_train, batch_size = 32, epochs = 100)
        self.ann.save(self.model_path)
        
    def test_predict(self):
        """predict the data with the trained neural network
        """
        self.output = (self.ann.predict(self.X_test) > 0.5 )
        print(np.concatenate((self.output.reshape(len(self.output),1), self.Y_test.reshape(len(self.Y_test),1)),1))
        
        
