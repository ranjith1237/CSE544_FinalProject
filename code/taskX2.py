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
		print("No Correlation")
	elif val > crit:
		print("Positively Correlated")
	else:
		print("Negatively Correlated")

def cal_pearson(A,B):
	print("###################################################")
	print("Performing pearson correlaton test for the Hypothesis")
	return (np.mean(A*B) - np.mean(A)*np.mean(B))/(np.std(A)*np.std(B))

if __name__ == '__main__':
	crit = 0.5
	
	path="../data/X_data"
	cols=["Crashes","Cases"]
	peace = pearson_correlaton()


	start_date = "2020-05-01"
	end_date = "2020-06-31"
	collisions = peace.get_data(path, cols[0], start_date, end_date)
	cases = peace.get_data(path, cols[1], start_date, end_date)
	gg = cal_pearson(collisions, cases)
	print("Sxy-value => ", gg)
	check_correlation(gg, crit)

	start_date = "2021-01-01"
	end_date = "2021-01-31"
	collisions = peace.get_data(path, cols[0], start_date, end_date)
	cases = peace.get_data(path, cols[1], start_date, end_date)
	gg = cal_pearson(collisions, cases)
	print("Sxy-value => ", gg)
	check_correlation(gg, crit)

	

