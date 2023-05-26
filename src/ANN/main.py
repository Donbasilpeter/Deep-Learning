from ANN import ANN

if __name__ == "__main__":
    
    data_path = './Data/Churn_Modelling.csv' #path for the data
    deep_network = ANN(data_path)
    deep_network.preprocess()
    deep_network.neural_network()
    
    
    