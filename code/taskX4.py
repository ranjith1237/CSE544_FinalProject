import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.stats import poisson, geom, binom
import os

class kstest_forX:
	def get_data(self, path, col, start_date, end_date):
		path=os.path.join(path,"X_Processed_Final.csv")
		df = pd.read_csv(path)
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_data = df.loc[between_two_dates]
		test_data=filtered_data[col].values
		return test_data

	def check_2sample(self, data1, data2):
		H_0 = ['We say that our dataset','follow','distribution']
		#In 2 sample we directly find the dmax(max distance between the CDFS of state1 and state2 data)
		dmax = find_dmax(data1, data2)
		print("dmax value => ", dmax)
		#perform hypothesis testing with the dmax we got and the critical value
		#hypothesis_testing(dmax, self.threshold, H_0, "data1", "data2")
		#plotting the 2CDFS for reference
		plot_2CDFs(data1, data2)
		return dmax

def plot_2CDFs(a, b):
	a = np.sort(a);
	#print(a)
	y_a = eCDF(a)
	b = np.sort(b)
	#print(b)
	y_b = eCDF(b)
	plt.figure('eCDF')
	plt.plot(a, y_a ,'-b',label='eCDF of Crashes')
	plt.plot(b, y_b ,'-r',label='eCDF of Cases')
	# plt.plot([age,age],[y_M[X_M.index(age)],y_F[X_F.index(age)]])
	plt.xlabel("x")
	plt.ylabel('Pr[X<=x]')
	plt.legend()
	plt.show()

def eCDF(data):
	# data = data/np.sum(data)
	# return np.cumsum(data)
	res = []
	for i in data:
		res.append(np.count_nonzero(i>=data)/len(data))
	return np.array(res)

def eCDF2(data1, data2):
	res = []
	for i in data1:
		res.append(np.count_nonzero(i>=data2)/len(data2))
	return np.array(res)

def find_dmax(data, param):
	# print("data ", data)
	# print("param ", param)
	x = eCDF(data)
	data2 = param
	ecdf_lb1 = eCDF2((data2 - 0.1), data)
	#print(ecdf_lb1)
	ecdf_lb2 = eCDF2((data2 - 0.1), data2)
	ecdf_rb1 = eCDF2((data2 + 0.1), data)
	ecdf_rb2 = eCDF2((data2 + 0.1), data2)
	max1 = np.max(np.abs(ecdf_lb1 - ecdf_lb2))
	max2 = np.max(np.abs(ecdf_rb1 - ecdf_rb2))
	return max(max1, max2)

	# print("dmax x ", x)
	# print("dmax y ", y)
	return np.amax(np.abs(x-y))

def kstest(path, cols):
	#declare the class with the states and critical value
	ks1 = kstest_forX()

	#get the confirmed cases and deaths for both the states
	start_date = "2020-05-01"
	end_date = "2020-06-31"
	crashes = ks1.get_data(path, cols[0], start_date, end_date)
	if((np.max(crashes)-np.min(crashes)) != 0): crashes = (crashes-np.min(crashes))/(np.max(crashes)-np.min(crashes))
	print(crashes)
	#crashes = (crashes-np.mean(crashes))/(np.std(crashes))
	cases = ks1.get_data(path, cols[1], start_date, end_date)
	if((np.max(cases)-np.min(cases)) != 0): cases = (cases-np.min(cases))/(np.max(cases)-np.min(cases))
	print(cases)
	#cases = (cases-np.mean(cases))/(np.std(cases))
	

	#2-sample
	print("###########################################################################################")
	print("Performing KS test 2-sample on two states to see whether they both follow same distribution")
	dmax = ks1.check_2sample(crashes, cases)
	if(dmax<0.05):
		print("Follows same distribution")
	else:
		print("Doesn't follow same distribution")


if __name__ == '__main__':
	path="../data/X_data"
	cols=["Crashes","Cases"]
	#perform kstest with the above states
	kstest(path, cols)
	

