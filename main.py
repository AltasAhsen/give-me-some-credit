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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from joblib import dump
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score, classification_report, average_precision_score

import warnings
warnings.filterwarnings("ignore")

# %% Model Evaluation Metrics Function
def collect_model_report(y_true, y_pred, y_pred_proba, model_name, target_names=None):
    # Main metrics
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    main_rows = []
    for label, scores in report_dict.items():
        if isinstance(scores, dict):
            main_rows.append({
                'Model': model_name,
                'Label': label,
                'Precision': round(scores.get('precision', 0), 3),
                'Recall': round(scores.get('recall', 0), 3),
                'F1-Score': round(scores.get('f1-score', 0), 3),
                'Support': int(scores.get('support', 0))
            })

    # Additional metrics
    additional_metrics = {
        'Model': model_name,
        'ROC AUC': round(roc_auc_score(y_true, y_pred_proba), 3),
        'PR AUC': round(average_precision_score(y_true, y_pred_proba), 3),
        'Balanced Accuracy': round(balanced_accuracy_score(y_true, y_pred), 3),
        'MCC': round(matthews_corrcoef(y_true, y_pred), 3),
    }

    return main_rows, additional_metrics

def save_combined_report(model_reports, additional_metrics_list, save_path):
    main_df = pd.DataFrame([row for model in model_reports for row in model])
    metrics_df = pd.DataFrame(additional_metrics_list)

    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        main_df.to_excel(writer, sheet_name='Classification Report', index=False)
        metrics_df.to_excel(writer, sheet_name='Additional Metrics', index=False)


# %% Data
df = pd.read_csv("data/cs-training.csv", usecols=lambda column: column != 'Unnamed: 0')
print(df.info()) # Missing values on MonthlyIncome and NumberOfDependents

# %% Defining variables
X=df.iloc[:,1:]
y=df["SeriousDlqin2yrs"]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in sss.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]    
    y_train, y_val = y[train_index], y[val_index]

X_train = X_train.copy()
X_val = X_val.copy()

# Checking if variables are independent
filtered_df = X_train[
    ( (X_train['NumberOfTime30-59DaysPastDueNotWorse'] == 0) &
    (X_train['NumberOfTimes90DaysLate'] > 0) ) |
    (X_train['NumberOfTime60-89DaysPastDueNotWorse'] > 0)
]
print(filtered_df)
# NumberOfTimes90DaysLate doesn't repeat on NumberOfTime30-59DaysPastDueNotWorse

plt.figure(figsize=(10,8))
sns.heatmap(X_train.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# %% Pointing out high correlation features

#high correlation on NumberOfTime30-59DaysPastDueNotWorse and NumberOfTimes90DaysLate
#high correlation on NumberOfTime30-59DaysPastDueNotWorse and NumberOfTime60-89DaysPastDueNotWorse
#high correlation on NumberOfTime60-89DaysPastDueNotWorse and NumberOfTimes90DaysLate

high_corr_columns = ['NumberOfTime30-59DaysPastDueNotWorse', 
                     'NumberOfTimes90DaysLate', 
                     'NumberOfTime60-89DaysPastDueNotWorse']

high_corr_table = X_train[high_corr_columns].describe()
print(high_corr_table)
# since the variance is fixed at one point, PCA is not a good idea
# I will create one variable for these three in a pipeline, so three variable won't dominate the prediction
# 90 Days Late should be more important than 30 Days late


# %% preprocessing components

class MergingColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y= None):
        return self
    def transform(self, X):
        X = X.copy()
        X["LatePaymentScore"] = (
            X['remainder__NumberOfTime30-59DaysPastDueNotWorse'] + 
            2 * X['remainder__NumberOfTime60-89DaysPastDueNotWorse'] + 
            3 * X['remainder__NumberOfTimes90DaysLate']
        )
        
        X.drop(columns=[
            'remainder__NumberOfTime30-59DaysPastDueNotWorse', 
            'remainder__NumberOfTimes90DaysLate', 
            'remainder__NumberOfTime60-89DaysPastDueNotWorse'
        ], inplace=True)
        return X


fillingValueTransformer = ColumnTransformer(
    transformers = [("monthlyIncome", SimpleImputer(strategy="median"), ["MonthlyIncome"]),
    ("numberOfDependents", SimpleImputer(strategy="most_frequent"), ["NumberOfDependents"])
    ], remainder='passthrough').set_output(transform='pandas')

#%% Checking covariance similarity for linear discrimination

temp_preprocessing_pipe = Pipeline([
    ("filling_the_missing_values", fillingValueTransformer),
    ("handling_high_correlation_cols", MergingColumns()),
    ("scaler", StandardScaler().set_output(transform='pandas')),
])

X_train_processed = temp_preprocessing_pipe.fit_transform(X_train)

cov_mat_0 = X_train_processed[y_train == 0].cov()
cov_mat_1 = X_train_processed[y_train == 1].cov()  

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

det_0 = np.linalg.det(cov_mat_0)
det_1 = np.linalg.det(cov_mat_1)
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 0: {det_0}")
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 1: {det_1}")

eig_vals_0, _ = np.linalg.eig(cov_mat_0)
eig_vals_1, _ = np.linalg.eig(cov_mat_1)

print(f"Eigenvalues for SeriousDlqin2yrs 0: {eig_vals_0}")
print(f"Eigenvalues for SeriousDlqin2yrs 1: {eig_vals_1}")

# %% QDA before SMOTE
qda_pipeline = ImbPipeline([
    ("filling_the_missing_values", fillingValueTransformer),
    ("handling_high_correlation_cols", MergingColumns()),
    ("scaler", StandardScaler().set_output(transform='pandas')),
    ("qda", QuadraticDiscriminantAnalysis())
])

qda_pipeline.fit(X_train, y_train)

y_pred = qda_pipeline.predict(X_val)
y_pred_proba = qda_pipeline.predict_proba(X_val)[:, 1]
qda_b_main, qda_b_add = collect_model_report(y_val, y_pred, y_pred_proba, model_name="QDA-before-SMOTE", target_names=["Negative", "Pozitive"])

dump(qda_pipeline, 'qda_before_SMOTE.joblib')
# model = load('qda_before_SMOTE.joblib')

# %% Training Decision Tree before SMOTE

dt_pipeline = ImbPipeline([
    ("filling_the_missing_values", fillingValueTransformer),
    ("handling_high_correlation_cols", MergingColumns()),
    ("scaler", StandardScaler().set_output(transform='pandas')),
    ("dtc", DecisionTreeClassifier())
])

param_grid = {
    'dtc__max_depth': [3, 5, 10, 15, None],
    'dtc__min_samples_split': [2, 5, 10],
    'dtc__min_samples_leaf': [1, 2, 5],
    'dtc__criterion': ['gini', 'entropy']
}

gridCV = GridSearchCV(
    estimator=dt_pipeline,
    param_grid= param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs= 1
)

gridCV.fit(X_train, y_train)
best_dt_pipeline = gridCV.best_estimator_

y_pred = best_dt_pipeline.predict(X_val)
y_pred_proba = best_dt_pipeline.predict_proba(X_val)[:, 1]
dt_b_main, dt_b_add = collect_model_report(y_val, y_pred, y_pred_proba, model_name="dt-before-SMOTE", target_names=["Negative", "Pozitive"])

dump(best_dt_pipeline, 'dt_before_SMOTE.joblib')
# model = load('dt_before_SMOTE.joblib')

# %% SMOTE
smote = SMOTE(random_state=42)  
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train_processed, y_train)

print("Original class distribution:", Counter(y_train))
print("Resampled class distribution:", Counter(y_train_SMOTE))

#%% Checking covariance similarity for linear discrimination - SMOTE sonrası

cov_mat_0 = X_train_SMOTE[y_train_SMOTE == 0].cov()  
cov_mat_1 = X_train_SMOTE[y_train_SMOTE == 1].cov()  

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

frobenius_norm = np.linalg.norm(cov_mat_0 - cov_mat_1, ord='fro')
print(f"Frobenius Norm of Covariance Matrix Difference: {frobenius_norm}")

det_0 = np.linalg.det(cov_mat_0)
det_1 = np.linalg.det(cov_mat_1)
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 0: {det_0}")
print(f"Determinant of Covariance Matrix for SeriousDlqin2yrs 1: {det_1}")

eig_vals_0, _ = np.linalg.eig(cov_mat_0)
eig_vals_1, _ = np.linalg.eig(cov_mat_1)

print(f"Eigenvalues for SeriousDlqin2yrs 0: {eig_vals_0}")
print(f"Eigenvalues for SeriousDlqin2yrs 1: {eig_vals_1}")

#This leads to QDA

# %% Training a QDA after SMOTE
qda_pipeline = ImbPipeline([
    ("filling_the_missing_values", fillingValueTransformer),
    ("handling_high_correlation_cols", MergingColumns()),
    ("scaler", StandardScaler().set_output(transform='pandas')),
    ("smote", SMOTE(random_state=42)),
    ("qda", QuadraticDiscriminantAnalysis())
])

qda_pipeline.fit(X_train, y_train)

y_pred = qda_pipeline.predict(X_val)
y_pred_proba = qda_pipeline.predict_proba(X_val)[:, 1]
qda_a_main, qda_a_add = collect_model_report(y_val, y_pred, y_pred_proba, model_name="QDA-after-SMOTE", target_names=["Negative", "Pozitive"])

dump(qda_pipeline, 'qda_after_SMOTE.joblib')
# model = load('qda_after_SMOTE.joblib')

# %% Training Decision Tree after SMOTE

dt_pipeline = ImbPipeline([
    ("filling_the_missing_values", fillingValueTransformer),
    ("handling_high_correlation_cols", MergingColumns()),
    ("scaler", StandardScaler().set_output(transform='pandas')),
    ("smote", SMOTE(random_state=42)),
    ("dtc", DecisionTreeClassifier())
])

param_grid = {
    'dtc__max_depth': [3, 5, 10, 15, None],
    'dtc__min_samples_split': [2, 5, 10],
    'dtc__min_samples_leaf': [1, 2, 5],
    'dtc__criterion': ['gini', 'entropy']
}

gridCV = GridSearchCV(
    estimator=dt_pipeline,
    param_grid= param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs= 1
)

gridCV.fit(X_train, y_train)
best_dt_pipeline = gridCV.best_estimator_

y_pred = best_dt_pipeline.predict(X_val)
y_pred_proba = best_dt_pipeline.predict_proba(X_val)[:, 1]
dt_a_main, dt_a_add = collect_model_report(y_val, y_pred, y_pred_proba, model_name="dt-after-SMOTE", target_names=["Negative", "Pozitive"])
dump(best_dt_pipeline, 'dt_after_SMOTE.joblib')
# model = load('dt_after_SMOTE.joblib')

# %% SVM kernel selection
# Method-1 : Interpreting Features of Decision Tree

# 1. Dominance Ratio
class_counts = np.bincount(y_train)
dominance_ratio = class_counts.max() / class_counts.sum()

# 2. Gini Impurity
classes, counts = np.unique(y_train, return_counts=True)
probs = counts / counts.sum()
gini = 1 - np.sum(probs**2)

# 3. Tree Depth - best_dt_pipeline'dan doğru şekilde erişim
tree_depth = best_dt_pipeline.named_steps['dtc'].get_depth()

# 4. Complexity Score
n_leaves = best_dt_pipeline.named_steps['dtc'].get_n_leaves()
complexity_score = tree_depth * np.log(n_leaves)

print("Dominance_ratio: ", dominance_ratio, "\nGini: ", gini, "\nDepth: ", tree_depth, "\nComplexity: ", complexity_score)
# Dominance_ratio:  0.5, Gini:  0.5, Depth:  15 , Complexity:  51.986038541995896 : It's a deep tree which means-non-linear relationships

# Method-2: Interpret Dimension Reduction
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_processed)
# UMAP
umap_model = UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_train_processed)

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
target_names = [f"Class {i}" for i in np.unique(y)]

# #  PCA
axs[0].set_title("PCA")
for label in np.unique(y_train):
    axs[0].scatter(X_pca[y_train == label, 0], X_pca[y_train == label, 1], label=target_names[label], alpha=0.6)
axs[0].legend()

# # UMAP
axs[1].set_title("UMAP")
for label in np.unique(y_train):
    axs[1].scatter(X_umap[y_train == label, 0], X_umap[y_train == label, 1], label=target_names[label], alpha=0.6)
axs[1].legend()

plt.show()

# %% SVM
svc_pipeline = ImbPipeline([
    ("filling_the_missing_values", fillingValueTransformer),
    ("handling_high_correlation_cols", MergingColumns()),
    ("scaler", StandardScaler().set_output(transform='pandas')),
    ("smote", SMOTE(random_state=42)),
    ("svc", SVC(probability=True, kernel = "rbf"))
])
svc_pipeline.fit(X_train, y_train)

y_pred = svc_pipeline.predict(X_val)
y_pred_proba = svc_pipeline.predict_proba(X_val)[:, 1]
svm_main, svm_add = collect_model_report(y_val, y_pred, y_pred_proba, model_name="svm-after-SMOTE", target_names=["Negative", "Pozitive"])

# %%
save_combined_report([qda_b_main, dt_b_main, qda_a_main, dt_a_main, svm_main], [qda_b_add, dt_b_add, qda_a_add, dt_a_add, svm_add], "all_models_report.xlsx")
# %%
