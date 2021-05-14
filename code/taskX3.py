import pandas as pd
import numpy as np

class LinearRegression():
	def __init__(self,p):
		self.p=p

	def get_data(self,path):
		df=pd.read_csv(path)
		data=df[['confirmed_cases','motor_vehicle_collisions']].values
		return data

	def get_train_test_data(self,data):
		X,y=data[:,0],data[:,1]
		trainX=[]
		trainY=[]
		p=self.p
		for i in range(0,len(X)):
			if i+p<len(X):
				trainX.append(X[i:i+p])
				trainY.append(y[i+p])
		trainX = np.array(trainX)
		v=np.expand_dims(np.ones(len(trainX)),axis=1)
		trainX = np.hstack((trainX,v))
		print(trainX)
		trainY = np.array(trainY)
		return trainX,trainY

	def fit_model(self,X,y):
		beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
		return beta

	def predict_next_days(self,weights,test_data,days=7):
		testX,testY=self.get_train_test_data(test_data)
		predicted=[]
		for i in range(len(testX)):
			y=np.dot(testX[i],weights)
			predicted.append(y)
		print("ground_truth===>",testY)
		print("predicted==>",predicted)
		return predicted

if __name__ == "__main__":
	path="../data/Us_data/MVC2.csv"
	df=pd.read_csv(path)
	lr = LinearRegression(7)
	data=lr.get_data(path)
	train_data = data[:70]
	test_data = data[70:]
	X_train,Y_train=lr.get_train_test_data(train_data)
	beta=lr.fit_model(X_train,Y_train)
	predicted = lr.predict_next_days(beta,test_data,days=7)