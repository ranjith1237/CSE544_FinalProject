import numpy as np
import scipy.stats as stats

def get_QObs(o1, o2, e1, e2):
	a = ((e1 - o1)**2)/e1
	print("a value", a)
	b = ((e2 - o2)**2)/e2
	print("b value", b)
	return np.sum((a,b))

if __name__ == '__main__':
	# data_2019 = np.array([1,2,3,4,5])
	# data_2020 = np.array([2,3,4,5,6])
	data_2019 = np.array([40,30])
	data_2020 = np.array([100,50])
	#compute Q_obs
	##find the expectations
	total_years = np.sum((data_2019, data_2020), axis = 0)
	total_atts =  np.sum((data_2019, data_2020), axis = 1)

	data_2019E = total_years * (total_atts[0]/np.sum(total_atts))
	data_2020E = total_years * (total_atts[1]/np.sum(total_atts))

	print(data_2019E)
	print(data_2020E)

	Q_obs = get_QObs(data_2019, data_2020, data_2019E, data_2020E)
	print("Chi-Square stat", Q_obs)
	df = data_2020.shape[0] - 1

	#get the p-value from t-table
	crit = stats.chi2.ppf(q=0.95, df=1)
	print("critical value", crit)

	if Q_obs < crit:
		print('Independent')
	else:
		print('Not Independent')



