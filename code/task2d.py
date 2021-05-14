import numpy as np
import pandas as pd
import scipy.stats as stats 
from matplotlib import pyplot as plt

class BayesianInference:

    def __init__(self):
        pass

    def load_csv(self):
        #Applying on all data since March data has been removed as part of Tukey's rule
        data_AK=pd.read_csv('../data/State_data/AK.csv').to_numpy()
        data_AL=pd.read_csv('../data/State_data/AL.csv').to_numpy()
        return data_AK, data_AL
    
    def get_combineddata(self, data_AK, data_AL):
        data = data_AK
        data[:,1:2] += data_AL[:,1:2] 
        data[:,2:3] += data_AL[:,2:3] 
        return data

    def get_poisson_mme(self, data):
        return np.sum(data)/data.size

    def get_samples(self, data):
        sample1 = data[np.where(data == '2020-06-01')[0][0]:np.where(data == '2020-06-28')[0][0]+1, 2:3].flatten()
        sample2 = data[np.where(data == '2020-06-29')[0][0]:np.where(data == '2020-07-05')[0][0]+1, 2:3].flatten()
        sample3 = data[np.where(data == '2020-07-06')[0][0]:np.where(data == '2020-07-12')[0][0]+1, 2:3].flatten()
        sample4 = data[np.where(data == '2020-07-13')[0][0]:np.where(data == '2020-07-19')[0][0]+1, 2:3].flatten()
        sample5 = data[np.where(data == '2020-07-20')[0][0]:np.where(data == '2020-07-26')[0][0]+1, 2:3].flatten()
        return sample1, sample2, sample3, sample4, sample5

    def get_posterior_map(self, beta, sample1, sample2, sample3, sample4):

        xStart = 0
        xEnd = 30

        alpha1 = np.sum(sample1) + 1
        beta1 = sample1.size + 1/beta
        # x1 = np.linspace(stats.gamma.ppf(0.01, alpha1, scale= 1/beta1), stats.gamma.ppf(0.99, alpha1, scale= 1/beta1), 100)
        x1 = np.linspace(xStart,xEnd, 10000)
        y1 = stats.gamma.pdf(x1, alpha1, scale=1/beta1)
        map1 = (alpha1 - 1)/beta1
        plt.plot(x1, y1, "y-", label="5th Week MAP: " + str(map1)) 


        alpha2 = np.sum(sample1) + np.sum(sample2) +  1
        beta2 = sample1.size + sample2.size + 1/beta
        x2 = np.linspace(xStart,xEnd, 10000)
        y2 = stats.gamma.pdf(x2, alpha2, scale=1/beta2)
        map2 = (alpha2 - 1)/beta2
        plt.plot(x2, y2, "r-", label="6th Week MAP: " + str(map2)) 


        alpha3 = np.sum(sample1) + np.sum(sample2) + np.sum(sample3) + 1
        beta3 = sample1.size + sample2.size + sample3.size + 1/beta
        x3 = np.linspace(xStart,xEnd, 10000)
        y3 = stats.gamma.pdf(x3, alpha3, scale=1/beta3)
        map3 = (alpha3 - 1)/beta3
        plt.plot(x3, y3, "g-", label="7th Week MAP: " + str(map3)) 


        alpha4 = np.sum(sample1) + np.sum(sample2) + np.sum(sample3) + np.sum(sample4) +  1
        beta4 = sample1.size + sample2.size + sample3.size + sample4.size + 1/beta
        x4 = np.linspace(xStart,xEnd, 10000)
        y4 = stats.gamma.pdf(x4, alpha4, scale=1/beta4)
        map4 = (alpha4 - 1)/beta4
        plt.plot(x4, y4, "b-", label="8th Week MAP: " + str(map4)) 


        plt.legend()
        plt.show()



bayesian = BayesianInference()
data_AK, data_AL = bayesian.load_csv()
if data_AL.shape[0] < data_AK.shape[0]:
    data_AK = data_AK[:data_AL.shape[0],:]
else:
    data_AL = data_AL[:data_AK.shape[0],:]
data = bayesian.get_combineddata(data_AK, data_AL)
sample1, sample2, sample3, sample4, sample5 = bayesian.get_samples(data)
beta = bayesian.get_poisson_mme(sample1)
bayesian.get_posterior_map(beta, sample2, sample3, sample4, sample5)