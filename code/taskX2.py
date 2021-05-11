import numpy as np

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
	A = np.array([48,40,58,53,65,25,52,34,30,45])
	B = np.array([54,48,51,47,62,35,70,20,25,40])
	C = np.array([19,40,35,41,38,32,32,37,37,15])
	crit = 0.05
	
	l = cal_pearson(A,B)
	m = cal_pearson(A,C)
	n = cal_pearson(B,C)

	check_correlation(l, crit)
	check_correlation(m, crit)
	check_correlation(n, crit)

	

