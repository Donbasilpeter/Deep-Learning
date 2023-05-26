from ANN import ANN1
from ANN import ANN2

if __name__ == "__main__":

    while (1):
        user_input = input(
            "select the ANN to invoke :\n ANN1 > 1\n ANN2 > 2 \n exit > exit \n\n Your option :  ")
        if user_input == "1":  # code for ANN1
            data_path = './Data/Churn_Modelling.csv'  # path for the data
            model_path = './Model/ANN1'
            deep_network = ANN1.ANN(data_path, model_path)
            break

        elif user_input == "2":
            data_path = './Data/CombinedCyclePowerPlant.xlsx'  # path for the data
            model_path = './Model/ANN2'
            deep_network = ANN2.ANN(data_path, model_path)
            break
        elif user_input =="exit":
            break

        else:
            print("Please give a valid response...!")


