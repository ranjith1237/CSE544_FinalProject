#KS test
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, geom, binom

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
	data = data/np.sum(data)
	return np.cumsum(data)

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
		print(ecdf_lb1)
		ecdf_lb2 = eCDF2((data2 - 0.1), data2)
		ecdf_rb1 = eCDF2((data2 + 0.1), data)
		ecdf_rb2 = eCDF2((data2 + 0.1), data2)
		max1 = np.max(np.abs(ecdf_lb1 - ecdf_lb2))
		max2 = np.max(np.abs(ecdf_rb1 - ecdf_rb2))
		return max(max1, max2)

	print("dmax x ", x)
	print("dmax y ", y)
	return np.amax(np.abs(x-y))

def get_CDF(data, dist):
	if dist=="empirical":
		plot_CDF(data)
	elif dist=="poisson":
		lamda = data


if __name__ == '__main__':
	data1 = np.array([1,2,3,4,5])
	data2 = np.array([1,2,3,4,5])
	c = 0.05
	#KS Test
	data1 = np.sort(data1)
	data2 = np.sort(data2)
	#1-sample, Poisson
	H_0 = ['We say that our dataset','follow','distribution']
	#finding the parameters for poisson
	lamda = np.mean(data1)
	print(lamda)
	dmax = find_dmax(data2, lamda, "poisson")
	print(dmax)
	hypothesis_testing(dmax, c, H_0, "data2", "poisson")


	#1-sample, Geometric
	k = np.mean(data1)
	p = np.reciprocal(k) if k > 1 else np.reciprocal(k+1)
	print(p)
	dmax = find_dmax(data2, p, "geometric")
	print(dmax)
	hypothesis_testing(dmax, c, H_0, "data2", "geometric")

	#1-sample, Binomial
	n = data1.shape[0]
	p = np.mean(data1)/n
	print(n, p)
	dmax = find_dmax(data2, [n, p], "binomial")
	print(dmax)
	hypothesis_testing(dmax, c, H_0, "data2", "binomial")

	#2-sample,
	dmax = find_dmax(data1, data2, "2-sample")
	print(dmax)
	hypothesis_testing(dmax, c, H_0, "data1", "data2")

	#Permutation test