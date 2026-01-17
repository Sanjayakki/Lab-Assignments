from operator import index

import numpy as np
import pandas as pd

def load_data(file_name):
    return pd.read_excel(file_name,usecols='A:E')

def read_Data(data):
    selected_Data=data[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']]
    X=selected_Data.values.tolist()
    selected_Output=data['Payment (Rs)']
    Y=selected_Output.values.tolist()
    return np.array(X),np.array(Y)

def calculate_rank(Matrix):
    Feature_matrix=np.array(Matrix)
    return np.linalg.matrix_rank(Feature_matrix)

def Cost_of_each_product(Features,Output):
    inv=np.linalg.pinv(Features)
    ans=inv @ Output
    return list(ans.round(2))

def Classification(Output):
    if Output>200:
        return "RICH"
    else:
        return "POOR"
#A1

file_path = 'Lab Session Data.xlsx'
df = load_data(file_path)

matrix_Feature, vector_Output = read_Data(df)


rank = calculate_rank(matrix_Feature)
print(f"Dimensionality (Rank) of the vector space: {rank}")



costs = Cost_of_each_product(matrix_Feature, vector_Output)
print(f"Cost of Candies, Mangoes, Milk: {costs}")


df['Category'] = df['Payment (Rs)'].apply(Classification)

#A2

print(df[['Customer','Payment (Rs)', 'Category']].to_string(index=False))