import numpy as np 
import pandas as pd 
import math
import datetime

class xPreProcessing:

    def __init__(self):
        pass

    def process_csv(self):
        x_data=pd.read_csv('../data/X_data/Motor_Vehicle_Collisions.csv', usecols=['CRASH DATE']).to_numpy()
        dates = [datetime.datetime.strptime(ts, "%m/%d/%Y") for ts in x_data.flatten()]
        dates.sort()
        sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
        
        sorteddates = np.array(sorteddates)
        y2020_start = np.where(sorteddates == '2020-01-22')[0][0]
        y2021_start = np.where(sorteddates == '2021-04-04')[0][0]
        
        
        sorteddates = sorteddates[y2020_start:y2021_start+1]
        print(sorteddates)
        unique, counts = np.unique(sorteddates, return_counts=True)
        dateList = []
        for key, value in dict(zip(unique, counts)).items():
            temp = [key,value]
            dateList.append(temp)
        dateList = np.array(dateList)
        pd.DataFrame(dateList).to_csv('../data/X_data/X_Processed.csv', header=["Date", "Crashes"], index=False)

    def merge_csvs(self):
        us_data = pd.read_csv('../data/Us_data/US_confirmed.csv').to_numpy()
        us_data = us_data[np.where(us_data == 'NY')[0][0]][1:]
        x_data=pd.read_csv('../data/X_data/X_Processed.csv').to_numpy()
        us_modified = []
        us_modified.append(us_data[0])
        for ind, i in enumerate(us_data[1:]):
            if (i - us_data[ind]) < 0:
                us_modified.append(0)
            else:
                us_modified.append(i - us_data[ind])
        us_modified = np.array(us_modified)
        final_data= np.hstack([x_data, us_modified.reshape(-1,1)])
        pd.DataFrame(final_data).to_csv('../data/X_data/X_Processed_Final.csv', header=["Date", "Crashes", "Cases"], index=False)
preProcess = xPreProcessing()
# preProcess.process_csv()
preProcess.merge_csvs()
