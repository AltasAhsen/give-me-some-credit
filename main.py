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
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("data/cs-training.csv", usecols=lambda column: column != 'Unnamed: 0')
print(df.info()) # Missing values on MonthlyIncome and NumberOfDependents

#%% Filling the missing values
monthly_income_mean = df['MonthlyIncome'].median()
df['MonthlyIncome'].fillna(monthly_income_mean, inplace=True)
number_of_dependents_mode = df['NumberOfDependents'].mode()[0]
df['NumberOfDependents'].fillna(number_of_dependents_mode, inplace=True)
print(df.info())

#%% Visualizing the data
for i in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[i], kde=True, bins=30, color='blue')
    plt.title(f'{i} Distribution')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show() # It is an unbalanced dataset. Around 140.000 times 0 in SeriousDlqin2yrs and 20.000 times 1

for i in df.columns:
    plt.figure(figsize=(10, 6))
    sns.stripplot(df[i], alpha=0.7, color='blue')
    plt.title(f'{i} Distribution')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show()


for col in df.select_dtypes(include=['float', 'int']).columns:
    print(f"{col}:")
    print(df[col].describe())
    print("\n")


# %% Checking if variables are independent
filtered_df = df[
    (df['NumberOfTime30-59DaysPastDueNotWorse'] == 0) &
    (df['NumberOfTimes90DaysLate'] > 0) &
    (df['NumberOfTime60-89DaysPastDueNotWorse'] > 0)
]
print(filtered_df)
# NumberOfTimes90DaysLate deosn't repeat on NumberOfTime30-59DaysPastDueNotWorse

# %% Pointing out high correlation features
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

# %% Creating PCA for high correlation
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

components = pca.components_
feature_contributions = pd.DataFrame(components, columns=high_corr_columns)
feature_contributions_percentage = feature_contributions.abs() / feature_contributions.abs().sum(axis=1).values.reshape(-1, 1) * 100

print("PCA Component Contribution Percentages:")
print(feature_contributions_percentage)
# It means PC1 = %33 of NumberOfTime30-59DaysPastDueNotWorse + %33 of NumberOfTimes90DaysLate + %33 of NumberOfTime60-89DaysPastDueNotWorse
print(pca.explained_variance_ratio_)
# [0.99186925]

# %% Defining variables
x_train=df.iloc[:,1:]

y_train=df["SeriousDlqin2yrs"]
standard_x = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    standard_x, y_train, test_size=0.2, random_state=42
)

# %% Visualizing standardized dataframe

for i in standard_x.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(standard_x[i], kde=True, bins=30, color='blue')
    plt.title(f'{i} Distribution')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show()

for i in standard_x.columns:
    plt.figure(figsize=(10, 6))
    sns.stripplot(standard_x[i], alpha=0.7, color='blue')
    plt.title(f'{i} Distribution')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show()


for col in standard_x.select_dtypes(include=['float', 'int']).columns:
    print(f"{col}:")
    print(standard_x[col].describe())
    print("\n") 


# %% Decision Tree parameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid.fit(X_train_split, y_train_split)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)
#En iyi parametreler: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
#En iyi skor: 0.8506437822762966

# %% Training Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42, min_samples_leaf=1, min_samples_split=2, class_weight='balanced')
model.fit(X_train_split, y_train_split)
y_pred = model.predict(X_val_split)
f1 = f1_score(y_val_split, y_pred, average='weighted')
print(f"F1 score: {f1:.2f}")
#F1 score: 0.81
roc_auc = roc_auc_score(y_val_split, y_pred)
print(f"ROC AUC score: {roc_auc:.2f}")
# ROC AUC score: 0.78

fpr, tpr, thresholds = roc_curve(y_val_split, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="RAndom Guess (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

#%% Understanding the logic behind decision tree

importances = model.feature_importances_
print(importances)
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=X_train_split.columns)
plt.title("Feature Importance")
plt.show()

# [0.2339294  0.01779955 0.00325999 0.         0.0090155  0.0160985   0.         0.71989706]

tree_rules = export_text(
    model, 
    feature_names=list(X_train_split.columns)  # Özellik isimlerini belirt
)
print(tree_rules)
"""
|--- PC1 <= -0.03
|   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |--- NumberRealEstateLoansOrLines <= 1.31
|   |   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |   |--- age <= 0.35
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- age >  0.35
|   |   |   |   |   |--- class: 0
|   |   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |   |--- NumberOfOpenCreditLinesAndLoans <= 0.01
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- NumberOfOpenCreditLinesAndLoans >  0.01
|   |   |   |   |   |--- class: 0
|   |   |--- NumberRealEstateLoansOrLines >  1.31
|   |   |   |--- DebtRatio <= -0.17
|   |   |   |   |--- age <= -1.34
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- age >  -1.34
|   |   |   |   |   |--- class: 0
|   |   |   |--- DebtRatio >  -0.17
|   |   |   |   |--- NumberRealEstateLoansOrLines <= 3.08
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- NumberRealEstateLoansOrLines >  3.08
|   |   |   |   |   |--- class: 1
|   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |--- NumberOfOpenCreditLinesAndLoans <= 0.98
|   |   |   |   |--- age <= 0.22
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- age >  0.22
|   |   |   |   |   |--- class: 0
|   |   |   |--- NumberOfOpenCreditLinesAndLoans >  0.98
|   |   |   |   |--- age <= -0.26
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- age >  -0.26
|   |   |   |   |   |--- class: 0
|   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |--- age <= 0.49
|   |   |   |   |--- NumberOfOpenCreditLinesAndLoans <= 0.40
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- NumberOfOpenCreditLinesAndLoans >  0.40
|   |   |   |   |   |--- class: 1
|   |   |   |--- age >  0.49
|   |   |   |   |--- age <= 2.25
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- age >  2.25
|   |   |   |   |   |--- class: 0
|--- PC1 >  -0.03
|   |--- PC1 <= 0.09
|   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |--- PC1 <= 0.01
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |   |   |--- class: 1
|   |   |   |--- PC1 >  0.01
|   |   |   |   |--- age <= -0.93
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- age >  -0.93
|   |   |   |   |   |--- class: 1
|   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |--- PC1 <= 0.01
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |   |   |--- class: 1
|   |   |   |--- PC1 >  0.01
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |   |   |--- class: 1
|   |--- PC1 >  0.09
|   |   |--- PC1 <= 0.17
|   |   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |   |--- NumberOfOpenCreditLinesAndLoans <= -0.19
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- NumberOfOpenCreditLinesAndLoans >  -0.19
|   |   |   |   |   |--- class: 1
|   |   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |   |   |--- class: 1
|   |   |--- PC1 >  0.17
|   |   |   |--- RevolvingUtilizationOfUnsecuredLines <= -0.02
|   |   |   |   |--- PC1 <= 0.33
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- PC1 >  0.33
|   |   |   |   |   |--- class: 1
|   |   |   |--- RevolvingUtilizationOfUnsecuredLines >  -0.02
|   |   |   |   |--- PC1 <= 0.41
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- PC1 >  0.41
|   |   |   |   |   |--- class: 1
"""



#%% Checking covariance similarity for linear discrimination
cov_mat_0 = X_train_split[y_train_split == 0].cov()  
cov_mat_1 = X_train_split[y_train_split == 1].cov()  

print("Covariance Matrix for SeriousDlqin2yrs 0:\n", cov_mat_0, "\n")
print("Covariance Matrix for SeriousDlqin2yrs 1:\n", cov_mat_1, "\n")

plt.figure(figsize=(8, 6))
sns.heatmap(cov_mat_0, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Covariance Matrix HeatMap for SeriousDlqin2yrs 0')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cov_mat_1, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Covariance Matrix HeatMap for SeriousDlqin2yrs 1')
plt.show()
# This variance difference leads to QDA

# Methods to formulize this difference

frobenius_norm = np.linalg.norm(cov_mat_0 - cov_mat_1, ord='fro')
print(f"Frobenius Norm of Covariance Matrix Difference: {frobenius_norm}")
#Frobenius Norm of Covariance Matrix Difference: 7.748443365369917

det_0 = np.linalg.det(cov_mat_0)
det_1 = np.linalg.det(cov_mat_1)
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 0: {det_0}")
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 1: {det_1}")
#Determinant of Covariance Matrix for SeriousDlqin2yrs 0: 0.3784817787643834
# Determinant of Covariance Matrix for SeriousDlqin2yrs 1: 0.11852843919289058

eig_vals_0, _ = np.linalg.eig(cov_mat_0)
eig_vals_1, _ = np.linalg.eig(cov_mat_1)

print(f"Eigenvalues for SeriousDlqin2yrs 0: {eig_vals_0}")
print(f"Eigenvalues for SeriousDlqin2yrs 1: {eig_vals_1}")
#Eigenvalues for SeriousDlqin2yrs 0: [1.53642194 1.23659586 1.13759559 1.03374666 0.9277071  0.47606098, 0.53704806 0.71419704]
#Eigenvalues for SeriousDlqin2yrs 1: [8.07710132 2.0483028  1.22394463 0.20844883 0.18036884 0.72903867, 0.59836647 0.3568883 ]

# %%
