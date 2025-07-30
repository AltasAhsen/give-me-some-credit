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

# %% Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, classification_report
from scipy.stats import zscore

import warnings
warnings.filterwarnings("ignore")

# %% Model Evaluation Metrics Function
results_df = pd.DataFrame()
def modified_classification_report(model_name, y_val, y_prob, y_pred_proba, results_df = None):
    class_metrics = []
    for cls in [0, 1]:
        precision = precision_score(y_val, y_prob, pos_label=cls)
        recall = recall_score(y_val, y_prob, pos_label=cls)
        f1 = f1_score(y_val, y_prob, pos_label=cls)
        roc_auc = roc_auc_score(y_val == cls, y_pred_proba)  
        class_metrics.append({
            'Model': model_name,
            'Class': cls,
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1 Score': round(f1, 3),
            'ROC AUC': round(roc_auc, 3),
        })
    
    mcc = matthews_corrcoef(y_val, y_prob)
    balanced_acc = balanced_accuracy_score(y_val, y_prob)
    report = classification_report(y_val, y_prob, digits=3)
    
    general_metrics = {
        'Model': model_name,
        'Class': 'Overall',
        'Precision': None,  
        'Recall': None,     
        'F1 Score': None,   
        'ROC AUC': None,    
        'MCC': round(mcc, 3),
        'Balanced Acc': round(balanced_acc, 3),
        'Classification_Report': report
    }
    
    df_class_metrics = pd.DataFrame(class_metrics)
    df_general_metrics = pd.DataFrame([general_metrics])
    
    model_results_df = pd.concat([df_class_metrics, df_general_metrics], ignore_index=True)

    if results_df is None:
        results_df = model_results_df
    else:
        results_df = pd.concat([results_df, model_results_df], ignore_index=True)
    return results_df


# %% Data
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
# 90 Days Late should be more importnat than 30 Days late

df['LatePaymentScore'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] +
    2 * df['NumberOfTime60-89DaysPastDueNotWorse'] +
    3 * df['NumberOfTimes90DaysLate'] 
)
df.drop(columns=['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse'], inplace=True)

print(df['LatePaymentScore'].describe())

# %% Defining variables
scaler = StandardScaler()
X=df.iloc[:,1:]
y=df["SeriousDlqin2yrs"]
standard_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_index, val_index in sss.split(X, y):
    X_train, X_val = standard_X.iloc[train_index], standard_X.iloc[val_index]    
    y_train, y_val = y[train_index], y[val_index]

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
# Frobenius Norm of Covariance Matrix Difference: 7.120624199876019

det_0 = np.linalg.det(cov_mat_0)
det_1 = np.linalg.det(cov_mat_1)
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 0: {det_0}")
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 1: {det_1}")
# Determinant of Covariance Matrix for SeriousDlqin2yrs 0: 0.3729770508242348
# Determinant of Covariance Matrix for SeriousDlqin2yrs 1: 0.16995540178160862

eig_vals_0, _ = np.linalg.eig(cov_mat_0)
eig_vals_1, _ = np.linalg.eig(cov_mat_1)

print(f"Eigenvalues for SeriousDlqin2yrs 0: {eig_vals_0}")
print(f"Eigenvalues for SeriousDlqin2yrs 1: {eig_vals_1}")
# Eigenvalues for SeriousDlqin2yrs 0: [1.51156773 0.45073134 0.53409605 0.71433014 1.23327751 1.13620735 0.95944205 1.06728707]
# Eigenvalues for SeriousDlqin2yrs 1: [7.40258486 2.16513628 1.22790289 0.1951373  0.35782767 0.27484533 0.75276357 0.59777968]

# This variance difference leads to QDA

# %% QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_val)
y_pred_proba = qda.predict_proba(X_val)[:, 1]
results_df = modified_classification_report("qda-before-SMOTE", y_val, y_pred, y_pred_proba, results_df)

# %% Decision Tree parameters
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
# Best parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
# Best score: 0.8494308978342376

# %% Training Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42, min_samples_leaf=1, min_samples_split=2, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]
results_df= modified_classification_report("decision-tree-before-SMOTE", y_val, y_pred, y_pred_proba, results_df)

importances = model.feature_importances_
print(importances)
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=X_train.columns)
plt.title("Feature Importance")
plt.show()

# %% SMOTE uygulama
smote = SMOTE(random_state=42)  # Rastgelelik için bir seed belirleyebilirsiniz
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Original class distribution:", Counter(y_train))
print("Resampled class distribution:", Counter(y_train_resampled))

#%% Checking covariance similarity for linear discrimination
y_train_resampled = y_train_resampled.reset_index(drop=True)
X_train_resampled = X_train_resampled.reset_index(drop=True)

cov_mat_0 = X_train_resampled[y_train_resampled == 0].cov()  
cov_mat_1 = X_train_resampled[y_train_resampled == 1].cov()  

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
# Frobenius Norm of Covariance Matrix Difference: 7.158037685749023

det_0 = np.linalg.det(cov_mat_0)
det_1 = np.linalg.det(cov_mat_1)
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 0: {det_0}")
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 1: {det_1}")
# Determinant of Covariance Matrix for SeriousDlqin2yrs 0: 0.3729770508242348
# Determinant of Covariance Matrix for SeriousDlqin2yrs 1: 0.07801523023152504

eig_vals_0, _ = np.linalg.eig(cov_mat_0)
eig_vals_1, _ = np.linalg.eig(cov_mat_1)

print(f"Eigenvalues for SeriousDlqin2yrs 0: {eig_vals_0}")
print(f"Eigenvalues for SeriousDlqin2yrs 1: {eig_vals_1}")
# Eigenvalues for SeriousDlqin2yrs 0: [1.51156773 0.45073134 0.53409605 0.71433014 1.23327751 1.13620735 0.95944205 1.06728707]
# Eigenvalues for SeriousDlqin2yrs 1: [7.42781272 2.10688322 1.21392259 0.29231128 0.24458355 0.14007467 0.72263076 0.56746374]

#This leads to QDA

# %% Training a QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_resampled, y_train_resampled)
y_pred = qda.predict(X_val)
y_pred_proba = qda.predict_proba(X_val)[:, 1]
results_df = modified_classification_report("qda", y_val, y_pred, y_pred_proba, results_df)


# %% Decision Tree parameters
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

grid.fit(X_train_resampled, y_train_resampled)

print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)
# Best parameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_leaf': 5, 'min_samples_split': 2}
# Best score: 0.9539837789333863

# %% Training Decision Tree
tree = DecisionTreeClassifier(criterion='gini', max_depth=15, random_state=42, min_samples_leaf=5, min_samples_split=2, class_weight='balanced')
tree.fit(X_train_resampled, y_train_resampled)
y_pred = tree.predict(X_val)
y_pred_proba = tree.predict_proba(X_val)[:, 1]
results_df= modified_classification_report("decision-tree", y_val, y_pred, y_pred_proba, results_df)

#%% Understanding the logic behind decision tree
importances = tree.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_resampled.columns, 
    'importance': importances
})

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=X_train.columns)
plt.title("Feature Importance")
plt.show()

# %% SVM kernel selection
# Method-1 : Interpreting Features of Decision Tree

# 1. Dominance Ratio
class_counts = np.bincount(y_train_resampled)
dominance_ratio = class_counts.max() / class_counts.sum()

# 2. Gini Impurity
classes, counts = np.unique(y_train_resampled, return_counts=True)
probs = counts / counts.sum()
gini = 1 - np.sum(probs**2)

# 3. Tree Depth
tree_depth = tree.get_depth()

# 4. Complexity Score
n_leaves = model.get_n_leaves()
complexity_score = tree_depth * np.log(n_leaves)

print("Dominance_ratio: ", dominance_ratio, "\nGini: ", gini, "\nDepth: ", tree_depth, "\nComplexity: ", complexity_score)
# Dominance_ratio:  0.5, Gini:  0.5, Depth:  15 , Complexity:  51.986038541995896 : It's a deep tree which means-non-linear relationships

# Method-2: Interpret Dimension Reduction
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_resampled)
# UMAP
umap_model = UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_train_resampled)

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
target_names = [f"Class {i}" for i in np.unique(y)]

# #  PCA
axs[0].set_title("PCA")
for label in np.unique(y):
    axs[0].scatter(X_pca[y_train_resampled == label, 0], X_pca[y_train_resampled == label, 1], label=target_names[label], alpha=0.6)
axs[0].legend()

# # UMAP
axs[1].set_title("UMAP")
for label in np.unique(y):
    axs[1].scatter(X_umap[y_train_resampled == label, 0], X_umap[y_train_resampled == label, 1], label=target_names[label], alpha=0.6)
axs[1].legend()

plt.show()

# %% SVM
svm_model = SVC(kernel="rbf", random_state=42, probability=True)
svm_model.fit(X_train_resampled, y_train_resampled)
y_pred = svm_model.predict(X_val)
y_pred_proba = svm_model.predict_proba(X_val)[:, 1]
results_df= modified_classification_report("SVM", y_val, y_pred, y_pred_proba, results_df)

# %% Comparing Models
results_df
