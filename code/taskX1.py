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
		#print("filtered_data",filtered_data)
		test_data=filtered_data[col].values
		return test_data

def get_QObs(o,e):
	a = ((e - o)**2)/e
	#print("a value", a)
	return np.sum(a)

def check_chi_square_test(path, cols):
	chi1 = chi_square_test()
	start_date = "2020-03-14"
	end_date = "2020-04-13"
	a1 = chi1.get_data(path, cols[0], start_date, end_date)
	a2 = chi1.get_data(path, cols[1], start_date, end_date)
	# a3 = chi1.get_data(path, cols[0], "01-02-2021", "28-02-2021")
	# a4 = chi1.get_data(path, cols[1], "01-02-2021", "28-02-2021")

	# print(a1.shape)
	# print(a2.shape)
	# data1 = np.array([np.mean(a1), np.mean(a2)])
	# data2 = np.array([np.mean(a3), np.mean(a4)])

	#data = np.vstack([np.hstack([a1.reshape(-1,1),a2.reshape(-1,1)]), np.hstack([a3.reshape(-1,1),a4.reshape(-1,1)])])
	data = np.hstack(([a1.reshape(-1,1),a2.reshape(-1,1)]))
	print(data)
	#compute Q_obs
	##find the expectations
	total_months = np.sum(data, axis = 0)
	total_atts =  np.sum(data, axis = 1)

	dataE = []
	for i in range(len(total_atts)):
		dataE.append(total_months * (total_atts[i]/np.sum(total_atts)))
	# data1E = total_months * (total_atts[0]/np.sum(total_atts))
	# data2E = total_months * (total_atts[1]/np.sum(total_atts))

	#print(dataE)
	# print(data2E)

	Q_obs = get_QObs(data, dataE)
	print("Chi-Square stat", Q_obs)
	df = data.shape[0] - 1

	#get the p-value from t-table
	print("Degrees of freedom =>", df)
	crit = stats.chi2.ppf(q=0.95, df=df)
	print("critical value", crit)

	if Q_obs < crit:
		print('Independent')
	else:
		print('Not Independent')


if __name__ == '__main__':
	# data_2019 = np.array([1,2,3,4,5])
	# data_2020 = np.array([2,3,4,5,6])
	# data_2019 = np.array([40,30])
	# data_2020 = np.array([100,50])
	
	path="../data/X_data"
	cols=["Crashes","Cases"]
	check_chi_square_test(path, cols)
	



