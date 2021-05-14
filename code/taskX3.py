import pandas as pd
import numpy as np

class LinearRegression():
	def __init__(self,p):#number of parameters in Linear Regression
		self.p=p

	def get_data(self,path):# data for each day number of cases and crashes
		df=pd.read_csv(path)
		data=df[['Cases','Crashes']].values
		return data

	def get_train_test_data(self,data):#prep train data in the required shape (n,p+1)
		X,y=data[:,0],data[:,1]
		trainX=[]
		trainY=[]
		p=self.p
		for i in range(0,len(X)):
			if i+p<len(X):
				trainX.append(X[i:i+p])
				trainY.append(y[i+p])
		trainX = np.array(trainX)
		v=np.expand_dims(np.ones(len(trainX)),axis=1) # adding new dimesion for bias term
		trainX = np.hstack((trainX,v))
		trainY = np.array(trainY)
		return trainX,trainY

	def fit_model(self,X,y):# fitting model
		beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y) #obtaining weights using regression formula
		return beta

	def predict_next_days(self,weights,test_data,days=7):#predicting for next days crashes using last 3 days cases of covid
		testX,testY=self.get_train_test_data(test_data)
		predicted=[]
		for i in range(len(testX)):
			y=np.dot(testX[i],weights)
			predicted.append(y)
		print("ground_truth===>",testY)
		print("predicted==>",predicted)
		_error_se,_error_mape = self.errors(testY,np.array(predicted))#obtaining MSE and MAPE
		print("Mean square error and MAPE for next 7 days is",_error_se,_error_mape)
		return _error_se

	def errors(self,y_test,y_pred): # calculates Mean squared error and MAPE
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

if __name__ == "__main__":
	path="../data/X_data/X_Processed_Final.csv"
	df=pd.read_csv(path)
	days=3
	lr = LinearRegression(days)
	data=lr.get_data(path)
	train_data = data[53:74]
	test_data = data[84-days+1:92]
	#print(train_data,test_data)
	X_train,Y_train=lr.get_train_test_data(train_data)
	beta=lr.fit_model(X_train,Y_train)
	predicted = lr.predict_next_days(beta,test_data,days=7)