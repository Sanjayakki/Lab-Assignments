import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename):
    return pd.read_excel(filename, sheet_name='thyroid0387_UCI',usecols="A:AE")

def Identifying_datatypes_Exploring(data):
    data_clean=data.replace('?',np.nan)
    analysis_report=[]
    for col in data_clean.columns:
    
        series_numeric=pd.to_numeric(data_clean[col],errors='coerce')
        is_numeric=sum(series_numeric.notna())>0 and sum(series_numeric.notna())>len(data_clean)*0.5
        unique_count=data_clean[col].nunique()
        missing_count=sum(data_clean[col].isnull())
        data_type='unknown'
        encoding='none'
   
        if not is_numeric:
            if unique_count<=2:
                data_type='Nominal'
                encoding='Binary Mapping'
            else:
                data_type='Nominal'
                encoding='One-Hot Encoding'
       
        else:
            
            valid_vals=series_numeric.dropna()
            if (valid_vals %1 ==0).all():
                data_type='Discrete'
            else:
                data_type='Continuous'
            encoding='-'

        data_range = "-"
        mean_val = "-"
        variance_val = "-"
        outliers_count = "-"

        if is_numeric:
            
            valid_vals = series_numeric.dropna()

           
            min_v = valid_vals.min()
            max_v = valid_vals.max()
            data_range = f"{min_v:.2f} - {max_v:.2f}"

            
            mean_val = round(valid_vals.mean(), 4)
            variance_val = round(valid_vals.var(), 4)

            
            std_dev = valid_vals.std()
            if std_dev > 0:
                z_scores = (valid_vals - valid_vals.mean()) / std_dev
                outliers_count = (abs(z_scores) > 3).sum()
            else:
                outliers_count = 0

        
        analysis_report.append({
            'Attribute': col,
            'Data Type': data_type,
            'Encoding Scheme': encoding,
            'Missing Values': missing_count,
            'Outliers (Z>3)': outliers_count,
            'Range': data_range,
            'Mean': mean_val,
            'Variance': variance_val
        })

    return pd.DataFrame(analysis_report)


def Get_Binary_Vectors(data):

   
    data_binary = data.copy()
    binary_cols = []

    for col in data_binary.columns:
        
        unique_vals = set(data_binary[col].dropna().unique())

       
        if unique_vals.issubset({'f', 't'}):
            data_binary[col] = data_binary[col].map({'f': 0, 't': 1})
            binary_cols.append(col)
      
        elif unique_vals.issubset({'F', 'M'}):
            data_binary[col] = data_binary[col].map({'F': 0, 'M': 1})
            binary_cols.append(col)
        
        elif unique_vals.issubset({0, 1}):
            binary_cols.append(col)


    vec1 = data_binary.iloc[0][binary_cols].values
    vec2 = data_binary.iloc[1][binary_cols].values

    return vec1, vec2

def Calculate_JC_And_SMC(v1, v2):
    
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 1) & (v2 == 0))
    f10 = np.sum((v1 == 0) & (v2 == 1))

    
    denominator_jc = f11 + f01 + f10
    jc = f11 / denominator_jc if denominator_jc != 0 else 0.0

    
    denominator_smc = f00 + f01 + f10 + f11
    smc = (f11 + f00) / denominator_smc if denominator_smc != 0 else 0.0
    return jc, smc


def Calculate_Cosine_Similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)

def heatmap_plot_Similarity(data):

   
    subset = data.head(20).copy()

    
    binary_subset = subset.copy()
    binary_cols = []

    for col in binary_subset.columns:
        unique_vals = set(binary_subset[col].dropna().unique())
  
        if unique_vals.issubset({'f', 't'}):
            binary_subset[col] = binary_subset[col].map({'f': 0, 't': 1})
            binary_cols.append(col)
        elif unique_vals.issubset({'F', 'M'}):
            binary_subset[col] = binary_subset[col].map({'F': 0, 'M': 1})
            binary_cols.append(col)

    binary_matrix = binary_subset[binary_cols].values

    
    numeric_matrix = subset.select_dtypes(include=[np.number]).fillna(0).values

    n = 20
    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))
    cosine_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
           
            jc, smc = Calculate_JC_And_SMC(binary_matrix[i], binary_matrix[j])
            jc_matrix[i, j] = jc
            smc_matrix[i, j] = smc

          
            cos = Calculate_Cosine_Similarity(numeric_matrix[i], numeric_matrix[j])
            cosine_matrix[i, j] = cos

    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(jc_matrix, annot=False, cmap='YlGnBu', ax=axes[0])
    axes[0].set_title('Jaccard Coefficient (Binary)')
   
    sns.heatmap(smc_matrix, annot=False, cmap='YlGnBu', ax=axes[1])
    axes[1].set_title('Simple Matching Coeff (Binary)')
    
    sns.heatmap(cosine_matrix, annot=False, cmap='coolwarm', ax=axes[2])
    axes[2].set_title('Cosine Similarity (Numeric)')

    plt.tight_layout()
    plt.show()


def Impute_Missing_Values(data):

    data_clean = data.copy()
   
    data_clean.replace('?', np.nan, inplace=True)

    for col in data_clean.columns:
     
        if data_clean[col].isnull().sum() == 0:
            continue

        if data_clean[col].dtype == 'object':
            mode_value = data_clean[col].mode()[0]
            data_clean[col] = data_clean[col].fillna(mode_value)
        else:
            mean = data_clean[col].mean()
            std_deviation = data_clean[col].std()
            max_value = data_clean[col].max()

            
            if std_deviation == 0:
                data_clean[col] = data_clean[col].fillna(mean)
                continue
          
            z_score_max = (max_value - mean) / std_deviation

            if abs(z_score_max) > 3:
                median_val = data_clean[col].median()  
                data_clean[col] = data_clean[col].fillna(median_val)
            else:
                data_clean[col] = data_clean[col].fillna(mean)

    return data_clean


def Normalize_Attribute(data):
    numeric_data = data.select_dtypes(include=[np.number])

    normalized_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
    return normalized_data

file_name = 'Lab Session Data.xlsx'


data = load_data(file_name)
#A4
results = Identifying_datatypes_Exploring(data)
print(results.to_string(index=False))
#A5

vec1_bin, vec2_bin = Get_Binary_Vectors(data)
jc, smc = Calculate_JC_And_SMC(vec1_bin, vec2_bin)
print(f"Jaccard Coefficient (Binary): {jc:.4f}")
print(f"Simple Matching Coeff (Binary): {smc:.4f}")

#A6

numeric_df = data.select_dtypes(include=[np.number]).fillna(0)
vec1_num = numeric_df.iloc[0].values
vec2_num = numeric_df.iloc[1].values
cosine_sim = Calculate_Cosine_Similarity(vec1_num, vec2_num)
print(f"Cosine Similarity (Numeric): {cosine_sim:.4f}")

#A7

heatmap_plot_Similarity(data)

#A8

missing_before = data.replace('?', np.nan).isnull().sum().sum()
print(f"Total missing values before: {missing_before}")

clean_data = Impute_Missing_Values(data)


missing_after = clean_data.isnull().sum().sum()
print(f"Total missing values after:  {missing_after}")

#A9

normalized_df = Normalize_Attribute(clean_data)


print("Sample of Normalized Data (First 5 rows, Age column):")

if not normalized_df.empty:
    first_col = normalized_df.columns[0]
    print(normalized_df[first_col].head())
else:
    print("No numeric data found to normalize.")