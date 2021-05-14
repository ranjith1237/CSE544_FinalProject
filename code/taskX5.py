import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def eCDF(data):
	# data = data/np.sum(data)
	# return np.cumsum(data)
	res = []
	for i in data:
		res.append(np.count_nonzero(i>=data)/len(data))
	return np.array(res)

def load_csv():
    #This plot is just basically to get a high level overview whether the datasets are anyway related before we 
    #start up with the whole hypothesis testing.
    data=pd.read_csv('../data/X_data/X_Processed_Final.csv').to_numpy()
    dates = data[:,:1].flatten()
    x = np.arange(1,len(dates)+1)
    a = data[:,1:2].flatten()
    b = data[:,2:3].flatten()
    plt.figure('Crashes vs Cases')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(dates, a)
    ax1.title.set_text('Crashes')
    ax2.scatter(dates, b)
    ax2.title.set_text('Cases')
    ax1.xaxis.set_major_locator(MaxNLocator(5)) 
    ax2.xaxis.set_major_locator(MaxNLocator(5)) 
    plt.show()

load_csv()