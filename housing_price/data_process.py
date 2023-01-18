# import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
print(os.listdir("../housing_price"))

def load_csv(file):
    """
    Load the CSV form from housing.csv and split training, validation and test set
    Args:
        file: file path
    """
#################################################################################
#                                                                               #
#                             Read all three csv files                          #
#                                                                               #
#################################################################################
    data = pd.read_csv(file, header = 0)

    return data

def impute_process(feature):
    """
    This function is to help impute the missing data if needed
    Args:
        feature: pandas dataframe
    """
#############################################################################
#                                                                           # 
#      1.Clean/fill missing in numerical and categorical features           #
#                                                                           #
#############################################################################
    data = pd.DataFrame()
    categ_list = feature.select_dtypes(include=['object']).columns.tolist()
    num_list = feature.select_dtypes(exclude=['object']).columns.tolist()

    for num_features in num_list:
        median_value = feature[num_features].median()
        data[num_features] = feature[num_features].fillna(value=median_value) 

    for categ_features in categ_list:
        data[categ_features] = feature[categ_features].fillna("Others")
    return data

def outlier_process(data):
    """
    This function is to help remove outliers/abnormal data if needed
    Args:
        data: pandas dataframe
    """
#############################################################################
#                                                                           # 
#                1.Clean/fill outliers in numerical features                #
#                                                                           #
#############################################################################
    num_list = data.select_dtypes(exclude=['object']).columns.tolist()

    for num in num_list:
        # drop zeros
        data = data.loc[~(data[num] == 0)]
#         ''' Removing the Outliers '''
#         Q1 = np.percentile(data[num], 25, interpolation = 'midpoint')
#         Q3 = np.percentile(data[num], 75, interpolation = 'midpoint')
#         IQR = Q3 - Q1
#         # Above Upper bound
#         upper = data[num] >= (Q3+1.5*IQR)
#         # Below Lower bound
#         lower = data[num] <= (Q1-1.5*IQR)
#         data.drop(upper[0], inplace = True)
#         data.drop(lower[0], inplace = True)
        return data

def visual_corr(feature):
    """
    This function is to visualization corr if needed
    Args:
        feature: pandas dataframe
    """
#############################################################################
#                                                                           # 
#                 1.visualize data by taking corr function                  #
#                                                                           #
#############################################################################
    plt.figure(figsize=(12, 12))
    sns.heatmap(feature.corr(), annot=True)
    plt.savefig('heatmap.png')
    plt.close()
    
    sns.pairplot(feature)
    plt.savefig('pairplot.png')
    plt.close()


def clean_corr(data, high_corr_feature):
    """
    This function is to merge corr if needed
    Args:
        data: pandas dataframe
        high_corr_feature: list[list]
    """
#############################################################################
#                                                                           # 
#   1.take merge(PCA) or simple drop overlap feature to clean colinearity   #
#                                                                           #
#############################################################################
   
    for pair in high_corr_feature:
        feature_name = 'percentage_' + pair[0]
        data[feature_name] = data[pair[0]].div(data[pair[1]].values)
    
    return data

def norms(data):
    """
    This function is to help numeric data scale
    Args:
        data: pandas dataframe

    """

    num_list = data.select_dtypes(exclude=['object']).columns.tolist()
    min_max_scaler = MinMaxScaler(feature_range=(0.01, 1))
    x_scaled = min_max_scaler.fit_transform(data[num_list].values)
    df = pd.DataFrame(x_scaled, columns=num_list)
    return df

def encoding(data, df_norm):
    """
    This function is to help categorical feature to be used in ML. It is 
    One-Hot encoding
    Args:
        data: pandas dataframe

    """
#############################################################################
#                                                                           #
#      2.One-hot encoding the categorical features                          #
#                                                                           #
#############################################################################
    categ_list = data.select_dtypes(include=['object']).columns.tolist()
    num_list = data.select_dtypes(exclude=['object']).columns.tolist()

    one_hot_encoded_data = pd.get_dummies(data, columns = categ_list)
    one_hot_encoded_data = one_hot_encoded_data.drop(columns=num_list, axis=1)
    df = df_norm.join(one_hot_encoded_data)
    return df


if __name__ == '__main__':
    housing = load_csv('housing.csv')
    housing_impute = impute_process(housing)
    housing_outlier = outlier_process(housing_impute)
    housing_corr = clean_corr(housing_outlier, [['total_bedrooms','total_rooms'],['households','population'], ['households','total_rooms']])
    housing_norm = norms(housing_corr)
    housing_encoding = encoding(housing_corr, housing_norm)
    housing_encoding.drop(['longitude', 'latitude', 'total_bedrooms', 'total_rooms', 'population', 'households', 'ocean_proximity_NEAR OCEAN'], axis = 1, inplace =True)
    # visual_corr(housing_corr)
    housing_encoding.to_csv('process.csv', index=False)

    # check value of processed data
    print(housing_encoding.head())
    print(housing_encoding.shape)
    print(housing_encoding.info())
    print(housing_encoding.isnull().sum())

    for column_name in housing_encoding.columns:
        column = housing_encoding[column_name]
        count = (column == 0).sum()
        print('Count of zeros in column ', column_name, ' is : ', count)
