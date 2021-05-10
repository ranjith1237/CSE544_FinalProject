import numpy as np
import pandas as pd

class MeanTests:

    def __init__(self):
        self.zalpha = 1.95996
        self.tone = 1.69726088
        self.ttwo = 1.67155277

    def load_csv(self):
        #Applying on all data since March data has been removed as part of Tukey's rule
        data_AK=pd.read_csv('../data/State_data/AK_original.csv').to_numpy()
        data_AL=pd.read_csv('../data/State_data/AL_original.csv').to_numpy()
        return data_AK, data_AL

    def get_months_data(self, data):
        feb_start = np.where(data == '2021-02-01')[0][0]
        feb_end = np.where(data == '2021-02-28')[0][0]
        feb_data = data[feb_start:feb_end+1,:]

        mar_start = np.where(data == '2021-03-01')[0][0]
        mar_end = np.where(data == '2021-03-31')[0][0]
        mar_data = data[mar_start:mar_end+1,:]

        return feb_data, mar_data

    def get_mle(self,data):
        return np.sum(data)/data.size
    
    def apply_one_walds(self, data):
        feb_data, mar_data = self.get_months_data(data)
        feb_data_count = feb_data[:,1:2].flatten()
        feb_data_deaths = feb_data[:,2:3].flatten()

        mar_data_count = mar_data[:,1:2].flatten()
        mar_data_deaths = mar_data[:,2:3].flatten()

        #count
        cnt_gt = self.get_mle(feb_data_count)
        cnt_pr = self.get_mle(mar_data_count)
        cnt_num = cnt_pr - cnt_gt
        cnt_den = np.sqrt(np.sum(mar_data_count))/mar_data_count.size
        cnt_w = abs(cnt_num/cnt_den)
        print(cnt_w)
        if cnt_w<=self.zalpha:
            print("Accept that means of the counts are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the counts are same between Feb'21 and Mar'21")

        #deaths
        deaths_gt = self.get_mle(feb_data_deaths)
        deaths_pr = self.get_mle(mar_data_deaths)
        deaths_num = deaths_pr - deaths_gt
        deaths_den = np.sqrt(np.sum(mar_data_deaths))/mar_data_deaths.size
        deaths_w = abs(deaths_num/deaths_den)
        print(deaths_w)
        if deaths_w<=self.zalpha:
            print("Accept that means of the deaths are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the deaths are same between Feb'21 and Mar'21")


    def apply_two_walds(self, data):
        feb_data, mar_data = self.get_months_data(data)
        feb_data_count = feb_data[:,1:2].flatten()
        feb_data_deaths = feb_data[:,2:3].flatten()

        mar_data_count = mar_data[:,1:2].flatten()
        mar_data_deaths = mar_data[:,2:3].flatten()

        #counts 
        cnt_gt = 0
        cnt_pr = self.get_mle(feb_data_count) - self.get_mle(mar_data_count)
        cnt_num = cnt_pr - cnt_gt
        cnt_den = np.sqrt(np.sum(feb_data_count)/np.square(feb_data_count.size) 
        + np.sum(mar_data_count)/np.square(mar_data_count.size))
        cnt_w = abs(cnt_num/cnt_den)
        print(cnt_w)
        if cnt_w<=self.zalpha:
            print("Accept that means of the counts are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the counts are same between Feb'21 and Mar'21")


        #deaths
        deaths_gt = 0
        deaths_pr = self.get_mle(feb_data_deaths) - self.get_mle(mar_data_deaths)
        deaths_num = deaths_pr - deaths_gt
        deaths_den = np.sqrt(np.sum(feb_data_deaths)/np.square(feb_data_deaths.size) 
        + np.sum(mar_data_deaths)/np.square(mar_data_deaths.size))
        deaths_w = abs(deaths_num/deaths_den)
        print(deaths_w)
        if deaths_w<=self.zalpha:
            print("Accept that means of the deaths are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the deaths are same between Feb'21 and Mar'21")

    def true_var(self, data):
        data_count = data[:,1:2].flatten()
        data_deaths = data[:,2:3].flatten()

        var_count = np.var(data_count, ddof=1)
        var_deaths = np.var(data_deaths, ddof=1)
        return var_count, var_deaths

    def apply_one_z(self, data, var_count, var_deaths):
        feb_data, mar_data = self.get_months_data(data)
        feb_data_count = feb_data[:,1:2].flatten()
        feb_data_deaths = feb_data[:,2:3].flatten()

        mar_data_count = mar_data[:,1:2].flatten()
        mar_data_deaths = mar_data[:,2:3].flatten()

        #count
        cnt_gt = self.get_mle(feb_data_count)
        cnt_pr = self.get_mle(mar_data_count)
        cnt_num = cnt_pr - cnt_gt
        cnt_den = np.sqrt(var_count/mar_data_count.size)
        cnt_z = abs(cnt_num/cnt_den)
        print(cnt_z)
        if cnt_z<=self.zalpha:
            print("Accept that means of the counts are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the counts are same between Feb'21 and Mar'21")

        #deaths
        deaths_gt = self.get_mle(feb_data_deaths)
        deaths_pr = self.get_mle(mar_data_deaths)
        deaths_num = deaths_pr - deaths_gt
        deaths_den = np.sqrt(var_deaths/mar_data_deaths.size)
        deaths_z = abs(deaths_num/deaths_den)
        print(deaths_z)
        if deaths_z<=self.zalpha:
            print("Accept that means of the deaths are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the deaths are same between Feb'21 and Mar'21")
    

    def apply_one_t(self, data):
        feb_data, mar_data = self.get_months_data(data)
        feb_data_count = feb_data[:,1:2].flatten()
        feb_data_deaths = feb_data[:,2:3].flatten()

        mar_data_count = mar_data[:,1:2].flatten()
        mar_data_deaths = mar_data[:,2:3].flatten()
        t_var_count, t_var_deaths = self.true_var(mar_data)

        #count
        cnt_gt = self.get_mle(feb_data_count)
        cnt_pr = self.get_mle(mar_data_count)
        cnt_num = cnt_pr - cnt_gt
        cnt_den = np.sqrt(t_var_count/mar_data_count.size)
        cnt_t = abs(cnt_num/cnt_den)
        print(cnt_t)
        if cnt_t<=self.tone:
            print("Accept that means of the counts are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the counts are same between Feb'21 and Mar'21")

        #deaths
        deaths_gt = self.get_mle(feb_data_deaths)
        deaths_pr = self.get_mle(mar_data_deaths)
        deaths_num = deaths_pr - deaths_gt
        deaths_den = np.sqrt(t_var_deaths/mar_data_deaths.size)
        deaths_t = abs(deaths_num/deaths_den)
        print(deaths_t)
        if deaths_t<=self.tone:
            print("Accept that means of the deaths are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the deaths are same between Feb'21 and Mar'21")

    
    def apply_two_t(self, data):
        feb_data, mar_data = self.get_months_data(data)
        feb_data_count = feb_data[:,1:2].flatten()
        feb_data_deaths = feb_data[:,2:3].flatten()

        mar_data_count = mar_data[:,1:2].flatten()
        mar_data_deaths = mar_data[:,2:3].flatten()

        t_feb_var_count, t_feb_var_deaths = self.true_var(feb_data)
        t_mar_var_count, t_mar_var_deaths = self.true_var(mar_data)
        #counts 
        cnt_gt = 0
        cnt_pr = self.get_mle(feb_data_count) - self.get_mle(mar_data_count)
        cnt_num = cnt_pr - cnt_gt
        cnt_den = np.sqrt((t_feb_var_count/feb_data_count.size)
        + (t_mar_var_count/mar_data_deaths.size))
        cnt_t = abs(cnt_num/cnt_den)
        print(cnt_t)
        if cnt_t<=self.ttwo:
            print("Accept that means of the counts are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the counts are same between Feb'21 and Mar'21")


        #deaths
        deaths_gt = 0
        deaths_pr = self.get_mle(feb_data_deaths) - self.get_mle(mar_data_deaths)
        deaths_num = deaths_pr - deaths_gt
        deaths_den = np.sqrt((t_feb_var_deaths/feb_data_deaths.size)
        + (t_mar_var_deaths/mar_data_deaths.size))
        deaths_t = abs(deaths_num/deaths_den)
        print(deaths_t)
        if deaths_t<=self.ttwo:
            print("Accept that means of the deaths are same between Feb'21 and Mar'21")
        else:
            print("Reject that means of the deaths are same between Feb'21 and Mar'21")

        

meanTest = MeanTests()
data_AK, data_AL = meanTest.load_csv()
print("######################################################")
print("For State AK")
print("######################################################")
print("")
print("One sample Wald's")
print("")
meanTest.apply_one_walds(data_AK)
print("")
print("Two sample Wald's")
print("")
meanTest.apply_two_walds(data_AK)
print("")
print("One sample Z")
print("")
var_count, var_deaths = meanTest.true_var(data_AK)
meanTest.apply_one_z(data_AK, var_count, var_deaths)
print("")
print("One sample T")
print("")
meanTest.apply_one_t(data_AK)
print("")
print("Two sample T")
print("")
meanTest.apply_two_t(data_AK)
print("")


print("######################################################")
print("For State AL")
print("######################################################")
print("")
meanTest.apply_one_walds(data_AL)
print("")
print("Two sample Wald's")
print("")
meanTest.apply_two_walds(data_AL)
print("")
print("One sample Z")
print("")
var_count, var_deaths = meanTest.true_var(data_AL)
meanTest.apply_one_z(data_AL, var_count, var_deaths)
print("")
print("One sample T")
print("")
meanTest.apply_one_t(data_AL)
print("")
print("Two sample T")
print("")
meanTest.apply_two_t(data_AL)
print("")
