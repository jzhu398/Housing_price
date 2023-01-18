from sklearn.model_selection import train_test_split
import data_process
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

data = data_process.load_csv('process.csv')
X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
X_train,X_test,y_train,y_test = X.iloc[X_train], X.iloc[X_test], X.iloc[y_train], X.iloc[y_test]

lin_reg = LinearRegression()
scores = []
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(X_train, y_train)):
    lin_reg.fit(X_train.iloc[train,:], y_train.iloc[train,:])
    score = lin_reg.score(X_train.iloc[test,:], y_train.iloc[test,:])
    scores.append(score)

pred = lin_reg.predict(X_test)
lin_rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print('RMSE: ', lin_rmse)
print('r2:_', r2)

# try LGBMRegressor for this problem
# # set parameters
# parameters = {
# 'n_estimators': 2500,
# 'num_leaves': 16,
# 'learning_rate': 0.05,
# 'colsample_bytree': 0.5,
# 'max_depth': None,
# 'reg_alpha': 0.0,
# 'reg_lambda': 0.0,
# 'min_split_gain': 0.0,
# 'min_child_weight': 5,
# 'boost_from_average': True,
# 'early_stopping_rounds': 200,
# 'huber_delta': 1.0,
# 'min_child_samples': 10,
# 'objective': 'regression_l2',
# 'subsample_for_bin': 50000,
# "metric": 'rmse'
# }
# lgbm = LGBMRegressor(**parameters)
# lgbm.fit(x, v, eval_set=[(x_test, v_test)])

# # predict
# y_pred = lgbm.predict(test_final, num_iteration=lgbm.best_iteration_)

# # save as csv file
# res = pd.DataFrame({'jobId': test_og.jobId, 'salary': y_pred})
# res.to_csv('test_salaries.csv', index=False)

# # feature importances 
# plot_feature_importance(lgbm.feature_importances_, x_test.columns, 'lightGBM')

# def plot_feature_importance(importance,names,model_type):
#     """
#     plot features importance
#     Note:
#         should consider permutation importance
#     Args:
#         importance: feature importance from the model
#         names: column names
#         model_type: str, model name
#     """
# #################################################################################
# #                                                                               #
# #                             Read all three csv files                          #
# #                                                                               #
# #################################################################################
#     #Create arrays from feature importance and feature names
#     feature_importance = np.array(importance)
#     feature_names = np.array(names)

#     #Create a DataFrame using a Dictionary
#     data={'feature_names':feature_names,'feature_importance':feature_importance}
#     fi_df = pd.DataFrame(data)

#     #Sort the DataFrame in order decreasing feature importance
#     fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

#     #Define size of bar plot
#     plt.figure(figsize=(10,8))
#     #Plot Searborn bar chart
#     sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#     #Add chart labels
#     plt.title(model_type + 'FEATURE IMPORTANCE')
#     plt.xlabel('FEATURE IMPORTANCE')
#     plt.ylabel('FEATURE NAMES')
#     plt.show()