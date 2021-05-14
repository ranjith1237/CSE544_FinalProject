#KS test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import factorial
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
		return test_data

	def check_poisson(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		#finding the parameters for poisson distribution using state1 data, mean gives the lamda value
		lamda = np.mean(state1_data)
		print(lamda)
		#finding the dmax(max distance between the CDFS of state2 data and poisson distribution)
		dmax = find_dmax(state2_data, lamda, "poisson")
		print("dmax value => ", dmax)
		#perform hypothesis testing with the dmax we got and the critical value
		hypothesis_testing(dmax, self.threshold, H_0, "state2_data", "poisson")
		#plotting the 2CDFS for reference
		plot_2CDFs(state2_data, lamda, col, "poisson")

	def check_geometric(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		#finding the parameters for geometric distribution using state1 data, 1/mean gives the probability for geometric dist
		k = np.mean(state1_data)
		p = np.reciprocal(k) if k > 1 else np.reciprocal(k+1)
		print(p)
		#finding the dmax(max distance between the CDFS of state2 data and geometric distribution)
		dmax = find_dmax(state2_data, p, "geometric")
		print("dmax value => ", dmax)
		#perform hypothesis testing with the dmax we got and the critical value
		hypothesis_testing(dmax, self.threshold, H_0, "state2_data", "geometric")
		#plotting the 2CDFS for reference
		plot_2CDFs(state2_data, p, col, "geometric")

	def check_binomial(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		#finding the parameters for binomial distribution using state1 data, mean/n gives the p value
		n = state1_data.shape[0]
		p = np.mean(state1_data)/n
		print(n, p)
		#finding the dmax(max distance between the CDFS of state2 data and binomial distribution)
		dmax = find_dmax(state2_data, [n, p], "binomial")
		print("dmax value => ", dmax)
		#perform hypothesis testing with the dmax we got and the critical value
		hypothesis_testing(dmax, self.threshold, H_0, "state2_data", "binomial")
		#plotting the 2CDFS for reference
		plot_2CDFs(state2_data, [n, p], col, "binomial")

	def check_2sample(self, state1_data, state2_data, col):
		H_0 = ['We say that our dataset','follow','distribution']
		#In 2 sample we directly find the dmax(max distance between the CDFS of state1 and state2 data)
		dmax = find_dmax(state1_data, state2_data, "2-sample")
		print("dmax value => ", dmax)
		#perform hypothesis testing with the dmax we got and the critical value
		hypothesis_testing(dmax, self.threshold, H_0, "state1_data", "state2_data")
		#plotting the 2CDFS for reference
		plot_2CDFs(state1_data, state2_data, col, "2-sample")

	def check_pstest(self, state1_data, state2_data, col):
		#getting lengths of two datasets before merging
		len1 = state1_data.shape[0]
		len2 = state2_data.shape[0]
		#calculating T_obsolute value from the means of two states
		T_obs = np.abs(np.mean(state1_data) - np.mean(state2_data))
		#combining the two states data
		combined_data = np.concatenate((state1_data, state2_data), axis=None)
		#declaring an empty array to add all the newly calculated differences
		T_arr = []
		#taking 1000 samples of random order from the combined data
		for i in range(1000):
			np.random.shuffle(combined_data)
			#calcualting and appending the differences of means(length of the datasets is also preserved here)
			T_i = np.abs(np.mean(combined_data[:len1]) - np.mean(combined_data[len1:]))
			#appending binary since we only need of cases where T_i > T_Obsolute
			T_arr.append(1 if T_i>T_obs else 0)
		#calculating the p-value from T_arr array
		p_value = np.sum(T_arr)/len(T_arr)
		print("p_value =>", p_value)
		if p_value < self.threshold:
			print("distribution of both the states are not the same for", col, "cases")
		else:
			print("distribution of both the states are the same for", col, "cases")


def plot_2CDFs(a, b, col, dist):
	a = np.sort(a)
	y_a = eCDF(a)
	if dist=="poisson":
		k = np.arange(0, np.max(a), 1)
		y_b = poisson.cdf(k, b)
		b = k
	elif dist=="geometric":
		k = np.arange(0, np.max(a), 1)
		y_b = geom.cdf(k, b)
		b = k
	elif dist=="binomial":
		k = np.arange(0, np.max(a), 1)
		y_b = binom.cdf(k, b[0], b[1])
		b = k
	elif dist=="2-sample":
		b = np.sort(b)
		y_b = eCDF(b)
		dist = "State2"
	
	# print("a is ", a)
	# print("y_a is ", y_a)
	# print("b is ", b)
	# print("y_b is ", y_b)
	# d,age = ks_test_values(M,F)
	plt.figure('eCDF')
	plt.plot(a, y_a ,'-b',label='eCDF of State1')
	plt.plot(b, y_b ,'-r',label='eCDF of ' + dist)
	# plt.plot([age,age],[y_M[X_M.index(age)],y_F[X_F.index(age)]])
	plt.xlabel(col)
	plt.ylabel('Pr[X<=x]')
	plt.grid()
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

def hypothesis_testing(dmax, c, H_0, dataset, dist):
	if dmax < c:
		print("Decision=> ",H_0[0],dataset,H_0[1],dist,H_0[2])
	else:
		print("Decision=> ",H_0[0],dataset,"doesn't",H_0[1],dist,H_0[2])

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

def kstest(path, states):
	#declare the class with the states and critical value
	ks1 = kstest_and_permtest(states[0], states[1], 0.05)

	#get the confirmed cases and deaths for both the states
	state1_confirmed_data = ks1.get_data(path, states[0], "confirmed")
	state2_confirmed_data = ks1.get_data(path, states[1], "confirmed")
	state1_deaths_data = ks1.get_data(path, states[0], "deaths")
	state2_deaths_data = ks1.get_data(path, states[1], "deaths")


	#1-sample, Poisson
	print("##############################################################################")
	print("Performing KS test 1-sample on two states guessing the distribution as poisson")
	ks1.check_poisson(state1_confirmed_data, state2_confirmed_data, "confirmed")
	ks1.check_poisson(state1_deaths_data, state2_deaths_data, "deaths")

	#1-sample, Geometric
	print("################################################################################")
	print("Performing KS test 1-sample on two states guessing the distribution as geometric")
	ks1.check_geometric(state1_confirmed_data, state2_confirmed_data, "confirmed")
	ks1.check_geometric(state1_deaths_data, state2_deaths_data, "deaths")

	#1-sample, Binomial
	print("###############################################################################")
	print("Performing KS test 1-sample on two states guessing the distribution as binomial")
	ks1.check_binomial(state1_confirmed_data, state2_confirmed_data, "confirmed")
	ks1.check_binomial(state1_deaths_data, state2_deaths_data, "deaths")

	#2-sample
	print("###########################################################################################")
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
	print("###########################################################################################")
	print("Performing permutation test on two states to see whether they both follow same distribution")
	pt1.check_pstest(state1_confirmed_data, state2_confirmed_data, "confirmed")
	pt1.check_pstest(state1_deaths_data, state2_deaths_data, "deaths")



if __name__ == '__main__':
	path="../data/State_data/"
	states=["AK","AL"]
	#perform kstest with the above states
	kstest(path, states)
	#perform permtest with the above states
	permtest(path, states)
	