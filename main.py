# %% [markdown]
# Features of Dataset
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

# %% [markdown]
# Model Results <br/> <br/>
# Before outlier correction <br/>
# Decision Tree: F1 = 0.81, ROC AUC = 0.78 <br/>
# QDA: F1 = 0.60, ROC AUC = 0.62

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import zscore

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

# %% Examining the data
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

# %% Visualizing the dataset
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

print(df.describe())

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

# %% Collecting High Correlation Variables in One Variable
high_corr_table = df[high_corr_columns].describe()
print(high_corr_table)

# since the variance is almost 0 in these variables, PCA is not a good idea
# I will create one variable for these three, so three variable won't dominate the prediction
df['LatePaymentScore'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] +
    df['NumberOfTimes90DaysLate'] +
    df['NumberOfTime60-89DaysPastDueNotWorse']
)
df.drop(columns=['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse'], inplace=True)

print(df['LatePaymentScore'].describe())

# %% Defining variables
scaler = StandardScaler()
x=df.iloc[:,1:]
y=df["SeriousDlqin2yrs"]

# in SeriousDlqin2yrs, there are 0 around 140000 and 1 around 20000. To fix this imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x, y)

standard_x = pd.DataFrame(scaler.fit_transform(X_resampled), columns=X_resampled.columns)

X_train, X_val, y_train, y_val = train_test_split(
    standard_x, y_resampled, test_size=0.2, random_state=42
)

# %% Decision Tree parameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3, 5, 10, 15, None],
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

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)
# Best parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 2}
# Best score: 0.942447830403931

# %% Training Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42, min_samples_leaf=5, min_samples_split=2, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1 score: {f1:.2f}")
roc_auc = roc_auc_score(y_val, y_pred)
print(f"ROC AUC score: {roc_auc:.2f}")
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

#%% Understanding the logic behind decision tree

importances = model.feature_importances_
print(importances)
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=X_train.columns)
plt.title("Feature Importance")
plt.show()

# [RevolvingUtilizationOfUnsecuredLines,age,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberRealEstateLoansOrLines,NumberOfDependents, PC1]
# [0.37774854 0.01392079 0.02087616 0.0118631  0.00957752 0.01584375 0.38289778 0.16727235]
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X_train.columns, filled=True, fontsize=10)
plt.show()


#%% Checking covariance similarity for linear discrimination
cov_mat_0 = X_train[y_train == 0].cov()  
cov_mat_1 = X_train[y_train == 1].cov()  

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

# Methods to formulize this difference
frobenius_norm = np.linalg.norm(cov_mat_0 - cov_mat_1, ord='fro')
print(f"Frobenius Norm of Covariance Matrix Difference: {frobenius_norm}")
#Frobenius Norm of Covariance Matrix Difference: 2.4780028826650056

det_0 = np.linalg.det(cov_mat_0)
det_1 = np.linalg.det(cov_mat_1)
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 0: {det_0}")
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 1: {det_1}")
#Determinant of Covariance Matrix for SeriousDlqin2yrs 0: 0.1702935509572893
#Determinant of Covariance Matrix for SeriousDlqin2yrs 1: 0.02177995430110475

eig_vals_0, _ = np.linalg.eig(cov_mat_0)
eig_vals_1, _ = np.linalg.eig(cov_mat_1)

print(f"Eigenvalues for SeriousDlqin2yrs 0: {eig_vals_0}")
print(f"Eigenvalues for SeriousDlqin2yrs 1: {eig_vals_1}")
#Eigenvalues for SeriousDlqin2yrs 0: [0.1225901  1.6247185  1.29983286 1.37053111 1.41124269 0.54276129 0.81660304 0.76730264]
#Eigenvalues for SeriousDlqin2yrs 1: [2.09125477 1.48575294 0.98816022 0.20491276 0.27724566 0.63600899 0.42417495 0.46284334]

# This variance difference leads to QDA
# %% QDA

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_val)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1 score: {f1:.2f}")
roc_auc = roc_auc_score(y_val, y_pred)
print(f"ROC AUC score: {roc_auc:.2f}")

# %% Outlier Optimization


# F1 for decision tree is 0.87, and 0.60 for QDA 
# ROC AUC for deicision tree is 0.87, and 0.62 for QDA
# outlier optimization is required

z_scores = np.abs((X_train - X_train.mean()) / X_train.std())
outliers_z = (z_scores > 3)
outlier_counts_z = outliers_z.sum(axis=1)
print("Z-Score Yöntemi ile Aykırı Değer Sayısı:")
print(outlier_counts_z)

# [RevolvingUtilizationOfUnsecuredLines,age,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberRealEstateLoansOrLines,NumberOfDependents, PC1]
# [0.2339294  0.01779955 0.00325999 0.         0.0090155  0.0160985   0.         0.71989706]

Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = (X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))
outlier_counts_iqr = outliers_iqr.sum()
print("\nIQR Yöntemi ile Aykırı Değer Sayısı:")
print(outlier_counts_iqr)

for col in X_train.columns:
    if col == "age":
      continue
    else:
      Q1 = X_train[col].quantile(0.25)
      Q3 = X_train[col].quantile(0.75)
      IQR = Q3 - Q1
      upper_limit = Q3 + 1.5 * IQR
      X_train[col] = np.where(X_train[col] > upper_limit, upper_limit, X_train[col])


z_scores = np.abs((X_train - X_train.mean()) / X_train.std())
outliers_z = (z_scores > 3)
outlier_counts_z = outliers_z.sum()
print("Z-Score Yöntemi ile Aykırı Değer Sayısı:")
print(outlier_counts_z)

Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = (X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))
outlier_counts_iqr = outliers_iqr.sum()
print("\nIQR Yöntemi ile Aykırı Değer Sayısı:")
print(outlier_counts_iqr)

for i in X_train.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(X_train[i], kde=True, bins=30, color='blue')
    plt.title(f'{i} Distribution')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.show()

