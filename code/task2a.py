import pandas as pd
import numpy as np
import os

#EWMA
class EWMA:
	def __init__(self,alpha=0.5):
		self.alpha=alpha
	
	def predict_next(self,X,days=7):
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

	def get_data(self,X_data):
		trainX=[]
		trainY=[]
		p=self.p
		for i in range(0,len(X_data)):
			if i+p<len(X_data):
				trainX.append(X_data[i:i+p])
				trainY.append(X_data[i+p])
		trainX = np.array(trainX)
		trainY = np.array(trainY)
		return trainX,trainY

	def fit_data(self,X,y):
		beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
		return beta

	def predict_next(self,weights,x,days=7):
		predicted=[]
		for day in range(days):
			y=np.dot(x,weights)
			x=np.append(x[1:],y)
			predicted.append(y)
		return predicted

def print_confirmed_cases(predicted_cases,predicted_deaths):
	for i in range(0,len(predicted_cases)):
		print(f"for the day {i} confirmed cases are {predicted_cases[i]} and fatalities are {predicted_deaths[i]}")

def prepare_data(path,state,col="confirmed"):
		path=os.path.join(path,state+".csv")
		df = pd.read_csv(path)
		start_date = "2020-08-01"
		end_date = "2020-08-21"
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_data = df.loc[between_two_dates]
		X_data=filtered_data[state+" "+col].values
		return X_data

def task1(path,countries,p=3):
	for state in states:
		print("#################")
		print(f" Auto Regression for state {state} in Aug last week with p={p}")
		ar1 = AutoRegression(p,state) #auto regression instance
		X_data = prepare_data(path,state,col="confirmed")
		trainX,trainY = ar1.get_data(X_data)
		weights = ar1.fit_data(trainX,trainY)
		predicted_cases = ar1.predict_next(weights,trainX[-1],days=7)
		
		ar2 = AutoRegression(p,state) #auto regression instance
		X_data = prepare_data(path,state,col="deaths")
		trainX,trainY = ar2.get_data(X_data)
		weights = ar2.fit_data(trainX,trainY)
		predicted_deaths = ar2.predict_next(weights,trainX[-1],days=7)
		
		
		print_confirmed_cases(predicted_cases,predicted_deaths)
		print("")

def task2(path,states,alpha=0.5):
	for state in states:
		print("#################")
		print(f" EWMA for state {state} in Aug last week with alpha={alpha}")
		ew1 = EWMA(alpha)
		X_data = prepare_data(path,state,col="confirmed")
		predicted_cases = ew1.predict_next(X_data,days=7)
		
		ew1 = EWMA(alpha)
		X_data = prepare_data(path,state,col="deaths")
		predicted_deaths = ew1.predict_next(X_data,days=7)
		
		print_confirmed_cases(predicted_cases,predicted_deaths)
		print("")

if __name__ == "__main__":
	path="../data/State_data/"
	states=["AK","AL"]
	task1(path,states,p=3) #AR with p=3
	task1(path,states,p=5)	#AR with p=5
	task2(path,states,alpha=0.5)
	task2(path,states,alpha=0.8)