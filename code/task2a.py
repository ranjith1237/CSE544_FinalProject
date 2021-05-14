import pandas as pd
import numpy as np
import os


#EWMA
class EWMA:
	def __init__(self,alpha=0.5):
		self.alpha=alpha
	
	def predict_next(self,X,days=7):#predicting for next day using EWMA till current day
		b=1-self.alpha
		predicted=[]
		pred=0
		prev=0
		for i in range(0,len(X)):
			prev=pred
			pred=(self.alpha*X[i]) + (b*prev)
		for day in range(days):
			predicted.append(pred)
			temp=pred
			pred = (self.alpha*pred)+(b*prev)
			prev=temp
			
		return predicted

#auto regression
class AutoRegression:
	def __init__(self,p,state="AK"):
		self.p = p
		self.state = state

	def get_data(self,X_data): # creating train matrix 
		trainX=[]
		trainY=[]
		p=self.p
		for i in range(0,len(X_data)):
			if i+p+1<len(X_data):
				trainX.append([1]+X_data[i:i+p+1]) # added new dimension 1
				trainY.append(X_data[i+p+1])
		trainX = np.array(trainX) # size of trainX is (n,p+1)
		trainY = np.array(trainY) # size of trainY is (n,1)
		return trainX,trainY

	def fit_data(self,X,y): #calculating weights for training points
		beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
		return beta

	def predict_next(self,weights,x,days=7):
		predicted=[]
		for day in range(days):# predict for next day using the preidcted values as reference
			y=np.dot(x,weights)
			x=np.append(x[2:],y)
			x=np.append(1,x)
			predicted.append(y)
		return predicted

def errors(y_test,y_pred): # calculates Mean squared error and MAPE
		n = len(y_test)
		SSE,APE = 0,0
		for i in range(n):
			SSE += (y_test[i]-y_pred[i])**2	#calculating SSE for each prediction
			if y_test[i]!=0:
				APE += (abs(y_test[i]-y_pred[i])*100)/y_test[i] #calculating APE for each prediction
			else:
				APE = 0
		Mean_SE = SSE/n	#MSE for all predictions
		Mean_APE = APE/n #MAPE for all predictions
		return Mean_SE,Mean_APE

def print_confirmed_cases(predicted_cases,predicted_deaths,Mean_SE_cases,Mean_APE_cases,Mean_SE_deaths,Mean_APE_deaths):
	#printing values for next 7 days
	# for i in range(0,len(predicted_cases)):
	# 	print(f"for the day {i} confirmed cases are {predicted_cases[i]} and fatalities are {predicted_deaths[i]}")
	print("#############################")
	print("MSE for cases==>",round(Mean_SE_cases,2))
	print("MSE for deaths==>",round(Mean_SE_deaths,2))
	print("MAPE for cases==>",round(Mean_APE_cases,2))
	print("MAPE for deaths==>",round(Mean_APE_deaths,2))

def prepare_data(path,state,col="confirmed"):
		path=os.path.join(path,state+".csv")
		df = pd.read_csv(path) #reading  csv file for state specific data
		start_train_date = "2020-08-01"
		end_train_date = "2020-08-21"
		start_test_date = "2020-08-22"
		end_test_date = "2020-08-28"

		after_start_date = df["Date"] >= start_train_date
		before_end_date = df["Date"] <= end_train_date
		between_two_dates = after_start_date & before_end_date
		filtered_data = df.loc[between_two_dates] # filtering cases and deaths for first 3 weeks
		X_train_data=filtered_data[state+" "+col].values

		after_start_date = df["Date"] >= start_test_date
		before_end_date = df["Date"] <= end_test_date
		between_two_dates = after_start_date & before_end_date
		filtered_data = df.loc[between_two_dates] # filtering cases and deaths for last week
		X_test_data=filtered_data[state+" "+col].values

		return X_train_data,X_test_data

def task1(path,states,p=3):
	for state in states:
		print("#################")
		print(f" Auto Regression for state {state} in Aug last week with p={p}")
		ar1 = AutoRegression(p,state) #auto regression instance
		X_train_data,X_test_data = prepare_data(path,state,col="confirmed")
		trainX,trainY = ar1.get_data(X_train_data)#gettting train data in the requried format
		weights = ar1.fit_data(trainX,trainY) # fitting model obtaining weights
		predicted_cases = ar1.predict_next(weights,trainX[-1],days=7) # predicting deaths
		Mean_SE_cases,Mean_APE_cases=errors(predicted_cases,X_test_data)#calcuating errors

		ar2 = AutoRegression(p,state) #auto regression instance
		X_train_data,X_test_data = prepare_data(path,state,col="deaths")
		trainX,trainY = ar2.get_data(X_train_data) #gettting train data in the required format
		weights = ar2.fit_data(trainX,trainY) # fitting model obtaining weights
		predicted_deaths = ar2.predict_next(weights,trainX[-1],days=7) # predicting deaths
		Mean_SE_deaths,Mean_APE_deaths=errors(predicted_deaths,X_test_data)#calcuating errors
		
		print_confirmed_cases(predicted_cases,predicted_deaths,Mean_SE_cases,Mean_APE_cases,Mean_SE_deaths,Mean_APE_deaths)
		print("")

def task2(path,states,alpha=0.5):
	for state in states:
		print("#################")
		print(f" EWMA for state {state} in Aug last week with alpha={alpha}")
		ew1 = EWMA(alpha) #EWMA regression instance
		X_train_data,X_test_data = prepare_data(path,state,col="confirmed")#gettting train ,test data in the requried format
		predicted_cases = ew1.predict_next(X_train_data,days=7) # predicting deaths
		Mean_SE_cases,Mean_APE_cases=errors(predicted_cases,X_test_data) #calcuating errors

		ew1 = EWMA(alpha) #EWMA regression instance
		X_train_data,X_test_data = prepare_data(path,state,col="deaths")#gettting train,test data in the requried format
		predicted_deaths = ew1.predict_next(X_train_data,days=7) # predicting deaths
		Mean_SE_deaths,Mean_APE_deaths=errors(predicted_deaths,X_test_data)#calcuating errors

		print_confirmed_cases(predicted_cases,predicted_deaths,Mean_SE_cases,Mean_APE_cases,Mean_SE_deaths,Mean_APE_deaths)
		print("")

if __name__ == "__main__":
	path="../data/State_data/"
	states=["AK","AL"]
	task1(path,states,p=3) #AR with p=3
	task1(path,states,p=5)	#AR with p=5
	task2(path,states,alpha=0.5) #EWMA(0.5)
	task2(path,states,alpha=0.8) #EWMA(0.8)