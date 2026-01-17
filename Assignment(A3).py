import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    return pd.read_excel(filename,sheet_name="IRCTC Stock Price",usecols='A:I')

def Calculate_MeanAndVariance(data):
    column_D=data['Price']
    X=np.array(column_D.values)
    return X.mean(),X.var()

def Own_Mean_and_Variance(data):
    column_D=data['Price']
    sum_of_observations=column_D.values.sum()
    No_of_observations=column_D.shape[0]
    Own_mean=sum_of_observations/No_of_observations
    observations_=pow(column_D.values-Own_mean,2)
    Own_variance=observations_.sum()/No_of_observations
    return Own_mean,Own_variance

def Compare_Accuracy(own_mean,Own_variance,package_mean,package_variance):
    if np.isclose(own_mean,package_mean) and np.isclose(Own_variance,package_variance):
        return True
    else:
        return False

def Measure_Complexity(data,function):
    total_time=0
    for i in range(10):
        start_time = time.time()
        function(data)
        end_time = time.time()
        total_time += end_time - start_time
    avg_time = total_time/10
    return avg_time

def Get_Wednesdat_mean(data):
    wedenesday_data=data[data['Day']=='Wed']
    return wedenesday_data['Price'].mean()

def Get_April_mean(data):
    april_data=data[data['Month']=='Apr']
    return april_data['Price'].mean()

def Compare_mean_of_sample_and_population(sample_mean,population_mean):
    if np.isclose(sample_mean,population_mean):
        return True
    return False

def Calculate_Loss_Probability(data):
    clean_chg=data['Chg%'].astype(str).str.replace('%','').astype(float)
    loss_stocks=clean_chg.apply(lambda x:x < 0)
    probability_of_loss=sum(loss_stocks)/len(clean_chg)
    return probability_of_loss

def Calculate_Profit_Probability_Wednesday(data):
    wednesday_data = data[data['Day'] == 'Wed']
    clean_chg = wednesday_data['Chg%'].astype(str).str.replace('%', '').astype(float)
    profit_stocks = clean_chg.apply(lambda x: x > 0)
    probability_of_profit = sum(profit_stocks) / len(data)
    return probability_of_profit

def Calculate_Conditional_ProfitProbaility_Wednesday(data):
    wednesday_data=data[data['Day']=='Wed']
    clean_chg=wednesday_data['Chg%'].astype(str).str.replace('%','').astype(float)
    profit_stocks=clean_chg.apply(lambda x:x > 0)
    probability_of_profit=sum(profit_stocks)/len(clean_chg)
    return probability_of_profit

def Plot_Chng_Vs_Day(data):
    plot_data=data.copy()
    plot_data['Chg%_Clean']=plot_data['Chg%'].astype(str).str.replace('%','').astype(float)
    order_ofDays=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    plot_data['Day']=plot_data['Day'].astype(str).str.strip()
    plot_data['Day']=pd.Categorical(plot_data['Day'],categories=order_ofDays,ordered=True)
    plot_data=plot_data.sort_values(by=['Day'])
    plt.figure(figsize=(10,6))
    plt.scatter(plot_data['Day'],plot_data['Chg%_Clean'],color='green',alpha=0.5)
    plt.title('Scatter plot of Chng data against the Day of the week')
    plt.ylabel('Price change Percentage')
    plt.xlabel('Day of the week')
    plt.grid(True)
    plt.show()

#A3
filename = 'Lab Session Data.xlsx'
data = load_data(filename)


pkg_mean, pkg_var = Calculate_MeanAndVariance(data)
own_mean, own_var = Own_Mean_and_Variance(data)
print(f"Package: Mean={pkg_mean:.4f}, Var={pkg_var:.4f}")
print(f"Own:     Mean={own_mean:.4f}, Var={own_var:.4f}")

if Compare_Accuracy(own_mean, own_var, pkg_mean, pkg_var):
        print("Accuracy Check: PASSED")
else:
        print("Accuracy Check: FAILED")


print("\n--- Complexity ---")
avg_time_pkg = Measure_Complexity(data, Calculate_MeanAndVariance)
avg_time_own = Measure_Complexity(data, Own_Mean_and_Variance)
print(f"Pkg Time: {avg_time_pkg:.6f}s")
print(f"Own Time: {avg_time_own:.6f}s")

print("\n--- Specific Means ---")
print(f"Wednesday Mean: {Get_Wednesdat_mean(data):.4f}")
print(f"April Mean:     {Get_April_mean(data):.4f}")

prob_loss = Calculate_Loss_Probability(data)
prob_profit_wed = Calculate_Profit_Probability_Wednesday(data)
prob_cond_wed = Calculate_Conditional_ProfitProbaility_Wednesday(data)

print(f"Loss Probability: {prob_loss:.4f}")
print(f"Profit on Wed (Joint): {prob_profit_wed:.4f}")
print(f"Profit given Wed (Conditional): {prob_cond_wed:.4f}")


Plot_Chng_Vs_Day(data)