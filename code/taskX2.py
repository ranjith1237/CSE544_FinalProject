import numpy as np
import pandas as pd
import os

class pearson_correlaton:
	def get_data(self, path, col, start_date, end_date):
		path=os.path.join(path,"X_Processed_Final.csv")
		df = pd.read_csv(path)
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_data = df.loc[between_two_dates]
		test_data=filtered_data[col].values
		return test_data

def check_correlation(val, crit):
	if np.abs(val) < crit:
		print("Independent")
	elif val > (1-crit):
		print("Correlated")
	else:
		print("Not Correlated")

def cal_pearson(A,B):
	return (np.mean(A*B) - np.mean(A)*np.mean(B))/(np.std(A)*np.std(B))

if __name__ == '__main__':
	# A = np.array([48,40,58,53,65,25,52,34,30,45])
	# B = np.array([54,48,51,47,62,35,70,20,25,40])
	# C = np.array([19,40,35,41,38,32,32,37,37,15])
	crit = 0.1
	
	path="../data/X_data"
	cols=["Crashes","Cases"]
	peace = pearson_correlaton()
	start_date = "2020-01-1"
	end_date = "2021-01-31"
	collisions = peace.get_data(path, cols[0], start_date, end_date)
	cases = peace.get_data(path, cols[1], start_date, end_date)

	gg = cal_pearson(collisions, cases)
	print("Sxy-value => ", gg)
	# l = cal_pearson(A,B)
	# m = cal_pearson(A,C)
	# n = cal_pearson(B,C)

	check_correlation(gg, crit)
	# check_correlation(l, crit)
	# check_correlation(m, crit)
	# check_correlation(n, crit)

	

