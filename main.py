# %% [markdown]
# # Features of Dataset
# 
# **id** <br/>
# **SeriousDlqin2yrs** : Whether the debtor has experienced serious financial difficulties in the last two years <br/>
# **RevolvingUtilizationOfUnsecuredLines** : ratio of the amount used by the borrower to the total unsecured credit limit <br/>
# **age** <br/>
# **NumberOfTime30-59DaysPastDueNotWorse** :Number of payments 30-59 days late in the last two years <br/>
# **DebtRatio** : Debt/Income <br/>
# **MonthlyIncome** <br/>
# **NumberOfOpenCreditLinesAndLoans** <br/>
# **NumberOfTimes90DaysLate** <br/>
# **NumberRealEstateLoansOrLines** <br/>
# **NumberOfTime60-89DaysPastDueNotWorse** <br/>
# **NumberOfDependents** <br/>

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("data/cs-training.csv", index_col=0)
print(df.info())

print("--------AFTER------")

# Missing values on MonthlyIncome and NumberOfDependents
monthly_income_mean = df['MonthlyIncome'].mean()
df['MonthlyIncome'].fillna(monthly_income_mean, inplace=True)
number_of_dependents_mode = df['NumberOfDependents'].mode()[0]
df['NumberOfDependents'].fillna(number_of_dependents_mode, inplace=True)
print(df.info())


# %%
filtered_df = df[
    (df['NumberOfTime30-59DaysPastDueNotWorse'] == 0) &
    (df['NumberOfTimes90DaysLate'] > 0) &
    (df['NumberOfTime60-89DaysPastDueNotWorse'] > 0)
]
print(filtered_df)
# NumberOfTimes90DaysLate deosn't repeat on NumberOfTime30-59DaysPastDueNotWorse

# %%
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#high correlation on NumberOfTime30-59DaysPastDueNotWorse and NumberOfTimes90DaysLate
#high correlation on NumberOfTime30-59DaysPastDueNotWorse and NumberOfTime60-89DaysPastDueNotWorse
#high correlation on NumberOfTime60-89DaysPastDueNotWorse and NumberOfTimes90DaysLate

high_corr_columns = ['NumberOfTime30-59DaysPastDueNotWorse', 
                     'NumberOfTimes90DaysLate', 
                     'NumberOfTime60-89DaysPastDueNotWorse']

#standardization of only these three component, for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[high_corr_columns])
#principal component
pca = PCA(n_components=1)
principal_component = pca.fit_transform(scaled_data)
#adding to table
df['PC1'] = principal_component
df.drop(columns=high_corr_columns, inplace=True)
#with this new variable(PC1), 3 column wont dominate the other features

print(df.head())
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# %%
components = pca.components_
feature_contributions = pd.DataFrame(components, columns=high_corr_columns)
feature_contributions_percentage = feature_contributions.abs() / feature_contributions.abs().sum(axis=1).values.reshape(-1, 1) * 100

print("PCA Component Contribution Percentages:")
print(feature_contributions_percentage)
# It means PC1 = %33 of NumberOfTime30-59DaysPastDueNotWorse + %33 of NumberOfTimes90DaysLate + %33 of NumberOfTime60-89DaysPastDueNotWorse

# %%
#standardization for whole dataframe
standardized_df = scaler.fit_transform(df)
#Covariance Matrix
cov_mat = pd.DataFrame(standardized_df).cov()
print(cov_mat)

#Graph of covariance matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cov_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Covariance Matrix HeatMap')
plt.show()

# %%
x_train=df.iloc[:,1:]
y_train=df["SeriousDlqin2yrs"]



