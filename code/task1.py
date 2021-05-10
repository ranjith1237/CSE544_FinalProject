import numpy as np 
import pandas as pd 
import math

class PreProcessing:
    def __init__(self):
        self.state_AK_dataset = np.array(list())
        self.state_AL_dataset = np.array(list())
        self.col_AK_list = ["Date","AK confirmed","AK deaths"]
        self.col_AL_list = ["Date","AL confirmed","AL deaths"]

    def load_state_csv(self):
        #loading the State Dataset to Numpy array
        self.state_AK_dataset=pd.read_csv('../data/State_data/1.csv', usecols=self.col_AK_list)
        self.state_AL_dataset=pd.read_csv('../data/State_data/1.csv', usecols=self.col_AL_list)
        return self.state_AK_dataset, self.col_AK_list, self.state_AL_dataset, self.col_AL_list

    def missing_state_values(self):
        #Checking if there are any missing values in the dataset
        return self.state_AK_dataset.isnull().sum().any() or self.state_AL_dataset.isnull().sum().any()

    def detect_outliers(self, dataset, columns):
        #Checking if there are any outliers in the dataset using Tukey's rule
        outliers = []
        #find indexes of Q3 and Q1
        q1 = math.ceil(len(dataset)/4)
        q3 = math.ceil(3*len(dataset)/4)
        #iterating over columns except Date
        for i in columns[1:]:
            data = dataset[i].to_numpy(dtype=object)
            #Sorting the column data
            #Challenge 1: All the values are not sorted as they should be
            #Hence sorted for finding the exact quartile ranges
            #Challenge 2: Few of the values are same and when sorted are replaced
            #up and below, so that might be causing few data misplaced
            data_sorted = np.argsort(data)

            #finding the values of Q3 and Q1
            Q1 = data[data_sorted[q1-1]]
            Q3 = data[data_sorted[q3-1]]

            #finding the bounds for Tukey's rule
            upper_bound = Q3 + (1.5)*(Q3-Q1)
            lower_bound = Q1 - (1.5)*(Q3-Q1)
            #Marking as an outlier even if one of both the columns is noted as outlier using Tukey's rule
            for j in np.nonzero(data > upper_bound)[0]:
                if j not in outliers:
                    outliers.append(j)
            for j in np.nonzero(data < lower_bound)[0]:
                if j not in outliers:
                    outliers.append(j)

        return outliers

    def remove_outliers(self, outliers, data, columns_data, filename):
        #Remove the outliers first from the dataset
        modified_data = data.drop(outliers)
        #Copy the Dataframe to a new csv file
        modified_data.to_csv('../data/State_data/'+filename+'.csv', columns=columns_data, index=False)

    def generate_original_csv(self, data, columns_data, filename):
        #Copy the Dataframe to a new csv file
        data.to_csv('../data/State_data/'+filename+'_original.csv', columns=columns_data, index=False)

            
        

preProcess = PreProcessing()
#Calling the below function to load the State data to numpy arrays along with columns.
data_AK, col_AK, data_AL, col_AL = preProcess.load_state_csv()
#Generate original csv for each state without removing outliers
preProcess.generate_original_csv(data_AK, col_AK, "AK")
preProcess.generate_original_csv(data_AL, col_AL, "AL")
#Calling the below function to check if there are any missing values in the data.
isMissing = preProcess.missing_state_values()
print("Is there are any missing values in the dataset?: " + str(isMissing))
print("")
#Calling the below function to check for the outliers using Tukey's rule.
print("######################################################")
print("For State AK")
print("######################################################")
print("")
outliers_AK = preProcess.detect_outliers(data_AK, col_AK)
print("Number of Outliers found = " + str(len(outliers_AK)))
print("")
print("######################################################")
print("For State AL")
print("######################################################")
print("")
outliers_AL = preProcess.detect_outliers(data_AL, col_AL)
print("Number of Outliers found = " + str(len(outliers_AL)))
print("")

#Removing the outliers and creating seperate csvs for both states
preProcess.remove_outliers(outliers_AK, data_AK, col_AK, "AK")
preProcess.remove_outliers(outliers_AL, data_AL, col_AL, "AL")