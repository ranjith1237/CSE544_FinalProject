#KS test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, geom, binom
import os

class kstest_and_permtest:
	def __init__(self, state1, state2, threshold):
		self.state1 = state1
		self.state2 = state2
		self.threshold = threshold

	def get_data(self, path, state, col):
		path=os.path.join(path,state+".csv")
		df = pd.read_csv(path)
		start_date = "2020-10-01"
		end_date = "2020-12-31"
		after_start_date = df["Date"] >= start_date
		before_end_date = df["Date"] <= end_date
		between_two_dates = after_start_date & before_end_date
		filtered_data = df.loc[between_two_dates]
		test_data=filtered_data[state+" "+col].values
		#print(X_data)
		#test_data = np.sort(test_data)
		#test_data = test_data/np.mean(test_data)
		return test_data

	def check_poisson(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		#finding the parameters for poisson
		lamda = np.mean(state1_data)
		print(lamda)
		dmax = find_dmax(state2_data, lamda, "poisson")
		print(dmax)
		hypothesis_testing(dmax, self.threshold, H_0, "state2_data", "poisson")

	def check_geometric(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		k = np.mean(state1_data)
		p = np.reciprocal(k) if k > 1 else np.reciprocal(k+1)
		print(p)
		dmax = find_dmax(state2_data, p, "geometric")
		print(dmax)
		hypothesis_testing(dmax, self.threshold, H_0, "state2_data", "geometric")

	def check_binomial(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		n = state1_data.shape[0]
		p = np.mean(state1_data)/n
		print(n, p)
		dmax = find_dmax(state2_data, [n, p], "binomial")
		print(dmax)
		hypothesis_testing(dmax, self.threshold, H_0, "state2_data", "binomial")

	def check_2sample(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		dmax = find_dmax(state1_data, state2_data, "2-sample")
		print(dmax)
		hypothesis_testing(dmax, self.threshold, H_0, "state1_data", "state2_data")

	def check_pstest(self, state1_data, state2_data, col):
		len1 = state1_data.shape[0]
		len2 = state2_data.shape[0]
		T_obs = np.abs(np.mean(state1_data) - np.mean(state2_data))
		combined_data = np.concatenate((state1_data, state2_data), axis=None)
		T_arr = []
		for i in range(1000):
			np.random.shuffle(combined_data)
			T_i = np.abs(np.mean(combined_data[:len1]) - np.mean(combined_data[len1:]))
			T_arr.append(1 if T_i>T_obs else 0)
		p_value = np.sum(T_arr)/len(T_arr)
		print("p_value is", p_value)
		if p_value < self.threshold:
			print("distribution of both the states are not the same for", col, "cases")
		else:
			print("distribution of both the states are the same for", col, "cases")



def plot_CDF(inp):
	inp = np.sort(inp)
	n = len(inp)
	x = [0]
	y = [0]
	#appending first element
	i=0
	count = 0
	while i<n:
		x.append(inp[i])
		y.append(y[len(y)-1])
		x.append(inp[i])
		check = inp[i]
		#print("i is ", i)
		while inp[i] == check:
			count+=1
			i=i+1
			#print("updating i to ", i)
			if i == n: break
		y.append(count/n)
	plt.plot(x, y, linestyle='-')
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

def hypothesis_testing(dmax, c, H_0, dataset, dist):
	if dmax < c:
		print(H_0[0],dataset,H_0[1],dist,H_0[2])
	else:
		print(H_0[0],dataset,"doesn't",H_0[1],dist,H_0[2])

def find_dmax(data, param, dist):
	# print("data ", data)
	# print("param ", param)
	x = eCDF(data)
	if dist=="poisson":
		y = poisson.cdf(data, param)
	elif dist=="geometric":
		y = geom.cdf(data, param)
	elif dist=="binomial":
		y = binom.cdf(data, param[0], param[1])
	elif dist=="2-sample":
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

def get_CDF(data, dist):
	if dist=="empirical":
		plot_CDF(data)
	elif dist=="poisson":
		lamda = data

def kstest(path, states):
	ks1 = kstest_and_permtest(states[0], states[1], 0.05)
	state1_confirmed_data = ks1.get_data(path, states[0], "confirmed")
	state2_confirmed_data = ks1.get_data(path, states[1], "confirmed")
	state1_deaths_data = ks1.get_data(path, states[0], "deaths")
	state2_deaths_data = ks1.get_data(path, states[1], "deaths")
	# print("state1_confirmed_data",state1_confirmed_data)
	# print("state1_confirmed_data shape", state1_confirmed_data.shape)
	# print("state2_confirmed_data",state2_confirmed_data)
	# print("state2_confirmed_data shape", state2_confirmed_data.shape)
	# plot_CDF(state1_confirmed_data)
	# plot_CDF(state2_confirmed_data)

	#1-sample, Poisson
	print("Performing KS test 1-sample on two states guessing the distribution as poisson")
	ks1.check_poisson(state1_confirmed_data, state2_confirmed_data, "confirmed")
	ks1.check_poisson(state1_deaths_data, state2_deaths_data, "deaths")


	#1-sample, Geometric
	print("Performing KS test 1-sample on two states guessing the distribution as geometric")
	ks1.check_geometric(state1_confirmed_data, state2_confirmed_data, "confirmed")
	ks1.check_geometric(state1_deaths_data, state2_deaths_data, "deaths")

	#1-sample, Binomial
	print("Performing KS test 1-sample on two states guessing the distribution as binomial")
	ks1.check_binomial(state1_confirmed_data, state2_confirmed_data, "confirmed")
	ks1.check_binomial(state1_deaths_data, state2_deaths_data, "deaths")

	#2-sample
	print("Performing KS test 2-sample on two states to see whether they both follow same distribution")
	ks1.check_2sample(state1_confirmed_data, state2_confirmed_data, "confirmed")
	ks1.check_2sample(state1_deaths_data, state2_deaths_data, "deaths")
	
def permtest(path, states):
	pt1 = kstest_and_permtest(states[0], states[1], 0.05)
	state1_confirmed_data = pt1.get_data(path, states[0], "confirmed")
	state2_confirmed_data = pt1.get_data(path, states[1], "confirmed")
	state1_deaths_data = pt1.get_data(path, states[0], "deaths")
	state2_deaths_data = pt1.get_data(path, states[1], "deaths")

	#permutation test
	pt1.check_pstest(state1_confirmed_data, state2_confirmed_data, "confirmed")
	pt1.check_pstest(state1_deaths_data, state2_deaths_data, "deaths")



if __name__ == '__main__':
	path="../data/State_data/"
	states=["AK","AL"]
	kstest(path, states)
	permtest(path, states)
	