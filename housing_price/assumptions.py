# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import data_process
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols

data = data_process.load_csv('process.csv')

### assumption 1: linear relationship
data_process.visual_corr(data)

### assumption 2: multivariate normality
for col in data.columns:
    sm.qqplot(data[col], line ='45')
    plt.savefig(col)

### assumption 3: no or little multicollinerarity (after model)
X = data.drop(['median_house_value'], axis=1)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

### assumption 4: residuals white noise (after model) and assumption 5: homoscedasticity
multi_model = ols('median_house_value ~ housing_median_age + median_income + percentage_total_bedrooms + percentage_households', data=data).fit()

print(multi_model.summary())
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(multi_model, 'median_income', fig=fig)
# sns.residplot(x='median_income', y='median_house_value', data=data) # check res variable by variable
# plt.show()
