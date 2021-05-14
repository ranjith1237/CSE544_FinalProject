import numpy as np
import pandas as pd
import scipy.stats as stats
import os

class chi_square_test():
	def get_data(self, path, col, start_date, end_date):
		path=os.path.join(path,"X_Processed_Final.csv")
		df = pd.read_csv(path)
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_data = df.loc[between_two_dates]
		test_data=filtered_data[col].values
		return test_data

def get_QObs(o,e):
	a = ((e - o)**2)/e
	return np.sum(a)

def check_chi_square_test(path, cols, start_date, end_date):
	print("###################################################")
	print("Performing chi square independence test for the Hypothesis")
	chi1 = chi_square_test()
	a1 = chi1.get_data(path, cols[0], start_date, end_date)
	a2 = chi1.get_data(path, cols[1], start_date, end_date)
	
	data = np.hstack(([a1.reshape(-1,1),a2.reshape(-1,1)]))
	#compute Q_obs
	##find the expectations
	total_months = np.sum(data, axis = 0)
	total_atts =  np.sum(data, axis = 1)

	dataE = []
	for i in range(len(total_atts)):
		dataE.append(total_months * (total_atts[i]/np.sum(total_atts)))

	Q_obs = get_QObs(data, dataE)
	print("Chi-Square stat", Q_obs)
	#degrees of freedom is (i-1)*(j-1) where i and j are no. of rows and columns
	df = data.shape[0] - 1

	#get the p-value from t-table
	print("Degrees of freedom =>", df)
	crit = stats.chi2.ppf(q=0.95, df=df)
	print("critical value", crit)

	#if Q obsoulute is less than critical value which is 0.05 we say Independent \
	if Q_obs < crit:
		print('Independent')
	else:
		print('Not Independent')


if __name__ == '__main__':
	
	path="../data/X_data"
	cols=["Crashes","Cases"]

	#Hypothesis testing 1 for the dates feb 13th to march 13th
	check_chi_square_test(path, cols, "2020-02-13", "2020-03-13")

	#Hypothesis testing 2 for the dates march 14th to april 13th
	check_chi_square_test(path, cols, "2020-03-14", "2020-04-13")
	



