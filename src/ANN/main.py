from ANN import ANN

if __name__ == "__main__":
    
    data_path = './Data/Churn_Modelling.csv' #path for the data
    model_path = './Model'
    deep_network = ANN(data_path,model_path)
    deep_network.test_predict()
    
    
    
    