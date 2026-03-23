import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)

import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)

print("All libraries imported successfully.")
# ── Upload passengers.csv ──────────────────────────────────────────────────
# METHOD A: Upload from local computer (run this cell and select the file)
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("passengers.csv (1).csv")
print("Dataset loaded. Shape:", df.shape)
# Class distribution
print("Survival counts:")
print(df["Survived"].value_counts())
print("\nSurvival percentages:")
print(df["Survived"].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
# Q11 — Bar plot: survival counts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Survival counts
survival_counts = df['Survived'].value_counts()
axes[0].bar(['Did Not Survive (0)', 'Survived (1)'],
            [survival_counts[0], survival_counts[1]],
            color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0].set_title('Survival Counts', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate([survival_counts[0], survival_counts[1]]):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 2: Survival rate by gender  (Q12)
gender_survival = df.groupby('Sex')['Survived'].mean() * 100
axes[1].bar(gender_survival.index, gender_survival.values,
            color=['#3498db', '#e91e8c'], edgecolor='black')
axes[1].set_title('Survival Rate by Gender', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(gender_survival.values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Plot 3: Survival rate by passenger class  (Q13)
class_survival = df.groupby('Pclass')['Survived'].mean() * 100
axes[2].bar([f'Class {c}' for c in class_survival.index], class_survival.values,
            color=['#f39c12', '#9b59b6', '#1abc9c'], edgecolor='black')
axes[2].set_title('Survival Rate by Passenger Class', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Survival Rate (%)')
axes[2].set_ylim(0, 100)
for i, v in enumerate(class_survival.values):
    axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_counts_gender_class.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nQ11: {survival_counts[1]/(survival_counts[0]+survival_counts[1])*100:.1f}% of passengers survived.")
print(f"Q12: Female survival rate: {gender_survival['female']:.1f}%  |  Male survival rate: {gender_survival['male']:.1f}%")
print(f"Q13: Class 1: {class_survival[1]:.1f}%  |  Class 2: {class_survival[2]:.1f}%  |  Class 3: {class_survival[3]:.1f}%")
# Q14 & Q15 — Age and Fare distributions by survival
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution  (Q14)
survivors = df[df['Survived'] == 1]['Age'].dropna()
non_survivors = df[df['Survived'] == 0]['Age'].dropna()

axes[0].hist(non_survivors, bins=30, alpha=0.6, color='#e74c3c', label='Did Not Survive', density=True)
axes[0].hist(survivors, bins=30, alpha=0.6, color='#2ecc71', label='Survived', density=True)
axes[0].set_title('Age Distribution by Survival', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Density')
axes[0].legend()

# Fare distribution  (Q15)
survivors_fare = df[df['Survived'] == 1]['Fare'].dropna()
non_survivors_fare = df[df['Survived'] == 0]['Fare'].dropna()

axes[1].hist(non_survivors_fare, bins=40, alpha=0.6, color='#e74c3c', label='Did Not Survive', density=True)
axes[1].hist(survivors_fare, bins=40, alpha=0.6, color='#2ecc71', label='Survived', density=True)
axes[1].set_title('Fare Distribution by Survival', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Fare')
axes[1].set_ylabel('Density')
axes[1].set_xlim(0, 300)
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_age_fare.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Mean age - Survivors: {survivors.mean():.1f}  |  Non-survivors: {non_survivors.mean():.1f}")
print(f"Mean fare - Survivors: {survivors_fare.mean():.1f}  |  Non-survivors: {non_survivors_fare.mean():.1f}")
# Optional: Correlation heatmap
plt.figure(figsize=(9, 7))
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
# Q2 — Missing value analysis
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print("Columns with missing values:")
print(missing_df.to_string())
# ── Working copy ──────────────────────────────────────────────────────────
df_clean = df.copy()

# Q6 — Feature Engineering (FamilySize, IsAlone, HasCabin)
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean['IsAlone']    = (df_clean['FamilySize'] == 1).astype(int)
df_clean['HasCabin']   = df_clean['Cabin'].notnull().astype(int)

# Q4 — Handle missing Age: fill with median per Sex and Pclass (most informative)
df_clean['Age'] = df_clean.groupby(['Sex', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
# Fallback for any remaining NaN
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

# Q5 — Handle missing Embarked: fill with mode
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

# Drop irrelevant columns
df_clean.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

print("Shape after cleaning:", df_clean.shape)
print("Remaining missing values:", df_clean.isnull().sum().sum())
print("\nColumns remaining:", df_clean.columns.tolist())
# Q7 — Final feature set
X = df_clean.drop(columns=['Survived'])
y = df_clean['Survived']

numeric_features     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric features   :", numeric_features)
print("Categorical features:", categorical_features)
print("\nTotal features:", len(numeric_features) + len(categorical_features))
# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                          numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'),    categorical_features)
])

# First split: 80% temp + 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Second split: 75% train + 25% validation (of temp = 60%/20% overall)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
)

# Q10
print("=== Q10: Dataset Splits ===")
for name, ys in [('Train', y_train), ('Validation', y_valid), ('Test', y_test)]:
    print(f"{name:12s}: {len(ys):4d} samples | Survival rate: {ys.mean()*100:.1f}%")
# Q16 — Tune C
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
lr_val_scores = []

for c in C_values:
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=c, max_iter=1000, random_state=SEED))
    ])
    pipe.fit(X_train, y_train)
    score = f1_score(y_valid, pipe.predict(X_valid))
    lr_val_scores.append(score)
    print(f"C={c:7.3f}  |  Validation F1: {score:.4f}")

best_C = C_values[np.argmax(lr_val_scores)]
print(f"\nBest C = {best_C}  (Validation F1 = {max(lr_val_scores):.4f})")

# Plot
plt.figure(figsize=(8, 4))
plt.plot(range(len(C_values)), lr_val_scores, 'o-', color='steelblue', linewidth=2)
plt.xticks(range(len(C_values)), [str(c) for c in C_values])
plt.xlabel('Regularization Strength C')
plt.ylabel('Validation F1-Score')
plt.title('Logistic Regression: C vs Validation F1', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_tuning.png', dpi=150, bbox_inches='tight')
plt.show()
# Final LR model — train with best C and evaluate on test set
pipe_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=best_C, max_iter=1000, random_state=SEED))
])
pipe_lr.fit(X_train, y_train)

y_pred_lr   = pipe_lr.predict(X_test)
y_proba_lr  = pipe_lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
auc_lr = auc(fpr_lr, tpr_lr)

# Q17 — Metrics
print("=== Q17: Logistic Regression — Test Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC  : {auc_lr:.4f}")

# Q18 — Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Did Not Survive', 'Survived'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Logistic Regression — Confusion Matrix', fontweight='bold')

# ROC curve
axes[1].plot(fpr_lr, tpr_lr, color='steelblue', lw=2, label=f'LR (AUC={auc_lr:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve — Logistic Regression', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_cm_roc.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm_lr.ravel()
print(f"\nQ18: False Positives: {fp}  |  False Negatives: {fn}")
# Q19 — Feature coefficients
lr_clf = pipe_lr.named_steps['classifier']
ohe_features = list(pipe_lr.named_steps['preprocessor']
                    .named_transformers_['cat'].get_feature_names_out(categorical_features))
all_features = numeric_features + ohe_features
coeffs = pd.Series(lr_clf.coef_[0], index=all_features).sort_values()

plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coeffs.values]
coeffs.plot(kind='barh', color=colors, edgecolor='black')
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Q19: Logistic Regression — Feature Coefficients', fontweight='bold')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 3 POSITIVE (associated with survival):", coeffs.tail(3).index.tolist())
print("Top 3 NEGATIVE (associated with non-survival):", coeffs.head(3).index.tolist())
# Q20 — Tune k
k_values = [1, 3, 5, 7, 11, 15, 21]
knn_val_scores = []

for k in k_values:
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])
    pipe.fit(X_train, y_train)
    score = accuracy_score(y_valid, pipe.predict(X_valid))
    knn_val_scores.append(score)
    print(f"k={k:2d}  |  Validation Accuracy: {score:.4f}")

best_k = k_values[np.argmax(knn_val_scores)]
print(f"\nBest k = {best_k}  (Validation Acc = {max(knn_val_scores):.4f})")

# Q20 — Plot
plt.figure(figsize=(8, 4))
plt.plot(k_values, knn_val_scores, 'o-', color='darkorange', linewidth=2, markersize=7)
plt.axvline(best_k, color='green', linestyle='--', label=f'Best k={best_k}')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Validation Accuracy')
plt.title('KNN: k vs Validation Accuracy', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_tuning.png', dpi=150, bbox_inches='tight')
plt.show()
# Q22 — Final KNN evaluation
pipe_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=best_k))
])
pipe_knn.fit(X_train, y_train)

y_pred_knn  = pipe_knn.predict(X_test)
y_proba_knn = pipe_knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
auc_knn = auc(fpr_knn, tpr_knn)

print("=== Q22: KNN — Test Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_knn):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_knn):.4f}")
print(f"ROC AUC  : {auc_knn:.4f}")

# Confusion matrix
plt.figure(figsize=(5, 4))
cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(cm_knn, display_labels=['Did Not Survive', 'Survived']).plot(colorbar=False, cmap='Oranges')
plt.title('KNN — Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('knn_cm.png', dpi=150, bbox_inches='tight')
plt.show()
# Q24 — Tune SVM (linear and RBF kernels)
svm_results = []

# Linear kernel
for c in [0.1, 1, 10, 100]:
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', C=c, probability=True, random_state=SEED))
    ])
    pipe.fit(X_train, y_train)
    score = accuracy_score(y_valid, pipe.predict(X_valid))
    svm_results.append({'kernel': 'linear', 'C': c, 'gamma': '-', 'val_acc': score})
    print(f"Linear  C={c:5.1f}  |  Val Acc: {score:.4f}")

# RBF kernel
for c in [0.1, 1, 10, 100]:
    for gamma in ['scale', 'auto', 0.01, 0.1]:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=c, gamma=gamma, probability=True, random_state=SEED))
        ])
        pipe.fit(X_train, y_train)
        score = accuracy_score(y_valid, pipe.predict(X_valid))
        svm_results.append({'kernel': 'rbf', 'C': c, 'gamma': str(gamma), 'val_acc': score})

svm_df = pd.DataFrame(svm_results).sort_values('val_acc', ascending=False)
print("\nTop 10 SVM configurations:")
print(svm_df.head(10).to_string(index=False))

best_svm = svm_df.iloc[0]
print(f"\nBest SVM: kernel={best_svm['kernel']}, C={best_svm['C']}, gamma={best_svm['gamma']}, Val Acc={best_svm['val_acc']:.4f}")
# Q25 — Final SVM evaluation
gamma_val = best_svm['gamma']
try:
    gamma_val = float(gamma_val)
except ValueError:
    pass

pipe_svm = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(
        kernel=best_svm['kernel'],
        C=best_svm['C'],
        gamma=gamma_val if best_svm['kernel'] == 'rbf' else 'scale',
        probability=True,
        random_state=SEED
    ))
])
pipe_svm.fit(X_train, y_train)

y_pred_svm  = pipe_svm.predict(X_test)
y_proba_svm = pipe_svm.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
auc_svm = auc(fpr_svm, tpr_svm)

print("=== Q25: SVM — Test Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_svm):.4f}")
print(f"ROC AUC  : {auc_svm:.4f}")

# Q26 — ROC curves: SVM vs LR
plt.figure(figsize=(7, 5))
plt.plot(fpr_lr,  tpr_lr,  lw=2, color='steelblue',  label=f'Logistic Regression (AUC={auc_lr:.3f})')
plt.plot(fpr_svm, tpr_svm, lw=2, color='darkorange', label=f'SVM (AUC={auc_svm:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Q26: ROC Curves — LR vs SVM', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svm_lr_roc.png', dpi=150, bbox_inches='tight')
plt.show()
# Q28 — Tune Decision Tree
dt_results = []
for max_d in [None, 3, 5, 7, 10, 15]:
    for min_split in [2, 5, 10, 20]:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                max_depth=max_d,
                min_samples_split=min_split,
                random_state=SEED
            ))
        ])
        pipe.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipe.predict(X_train))
        val_acc   = accuracy_score(y_valid, pipe.predict(X_valid))
        dt_results.append({
            'max_depth': str(max_d),
            'min_samples_split': min_split,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

dt_df = pd.DataFrame(dt_results).sort_values('val_acc', ascending=False)
print("Top 10 Decision Tree configurations (by Validation Accuracy):")
print(dt_df.head(10).to_string(index=False))

best_dt = dt_df.iloc[0]
best_max_d     = None if best_dt['max_depth'] == 'None' else int(best_dt['max_depth'])
best_min_split = int(best_dt['min_samples_split'])
print(f"\nBest DT: max_depth={best_max_d}, min_samples_split={best_min_split}")
# Q29 — Overfitting: shallow vs deep tree
print("Q29 — Overfitting analysis: shallow vs deep tree")
print("-" * 55)
for depth_label, depth_val in [('Shallow (depth=3)', 3), ('Deep (depth=None)', None)]:
    p = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=depth_val, random_state=SEED))
    ])
    p.fit(X_train, y_train)
    tr = accuracy_score(y_train, p.predict(X_train))
    vl = accuracy_score(y_valid, p.predict(X_valid))
    print(f"{depth_label:25s}: Train={tr:.3f}  Val={vl:.3f}  Gap={tr-vl:.3f}")
# Q30 — Final DT evaluation
pipe_tree = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=best_max_d,
        min_samples_split=best_min_split,
        random_state=SEED
    ))
])
pipe_tree.fit(X_train, y_train)

y_pred_tree  = pipe_tree.predict(X_test)
y_proba_tree = pipe_tree.predict_proba(X_test)[:, 1]
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)
auc_tree = auc(fpr_tree, tpr_tree)

print("=== Q30: Decision Tree — Test Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_tree):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tree):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_tree):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_tree):.4f}")
print(f"ROC AUC  : {auc_tree:.4f}")
# Q31 — Decision Tree feature importances
dt_clf = pipe_tree.named_steps['classifier']
dt_importances = pd.Series(dt_clf.feature_importances_, index=all_features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
dt_importances.plot(kind='barh', color='teal', edgecolor='black')
plt.title('Q31: Decision Tree — Feature Importances', fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('dt_importances.png', dpi=150, bbox_inches='tight')
plt.show()
print("Top 3 features:", dt_importances.tail(3).index.tolist())
# Q32 — Tune Random Forest
rf_results = []
configs = [
    (50,  None, 'sqrt'), (50,  5,    'sqrt'), (100, None, 'sqrt'),
    (100, 5,    'log2'), (200, 10,   'sqrt'), (200, None, 'log2'),
    (100, 10,   'sqrt'), (200, 15,   'log2'), (50,  10,   'log2'),
]

for n_est, max_d, max_f in configs:
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            max_features=max_f, random_state=SEED
        ))
    ])
    pipe.fit(X_train, y_train)
    score = accuracy_score(y_valid, pipe.predict(X_valid))
    rf_results.append({'n_estimators': n_est, 'max_depth': str(max_d),
                       'max_features': max_f, 'val_acc': score})

rf_df = pd.DataFrame(rf_results).sort_values('val_acc', ascending=False)
print("Random Forest configurations (sorted by Validation Accuracy):")
print(rf_df.to_string(index=False))

best_rf = rf_df.iloc[0]
best_n_est = int(best_rf['n_estimators'])
best_rf_depth = None if best_rf['max_depth'] == 'None' else int(best_rf['max_depth'])
best_rf_feat  = best_rf['max_features']
print(f"\nBest RF: n_estimators={best_n_est}, max_depth={best_rf_depth}, max_features={best_rf_feat}")
# Q33 — Final RF evaluation
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_rf_depth,
        max_features=best_rf_feat,
        random_state=SEED
    ))
])
pipe_rf.fit(X_train, y_train)

y_pred_rf  = pipe_rf.predict(X_test)
y_proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = auc(fpr_rf, tpr_rf)

print("=== Q33: Random Forest — Test Metrics ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC  : {auc_rf:.4f}")
# Q34 — RF feature importances + comparison
rf_clf = pipe_rf.named_steps['classifier']
rf_importances = pd.Series(rf_clf.feature_importances_, index=all_features).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rf_importances.tail(10).plot(kind='barh', ax=axes[0], color='forestgreen', edgecolor='black')
axes[0].set_title('Random Forest — Top 10 Feature Importances', fontweight='bold')
axes[0].set_xlabel('Importance')

dt_importances.tail(10).plot(kind='barh', ax=axes[1], color='teal', edgecolor='black')
axes[1].set_title('Decision Tree — Top 10 Feature Importances', fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('rf_dt_importances.png', dpi=150, bbox_inches='tight')
plt.show()
# Q36 & Q37 — 5-Fold CV on best model (Random Forest)
X_cv = pd.concat([X_train, X_valid])
y_cv = pd.concat([y_train, y_valid])

pipe_cv = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_rf_depth,
        max_features=best_rf_feat,
        random_state=SEED
    ))
])

kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(pipe_cv, X_cv, y_cv, cv=kfold, scoring='accuracy')

print("=== Q36: 5-Fold Cross-Validation (Random Forest) ===")
for i, s in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {s:.4f}")
print(f"\nMean CV Accuracy : {cv_scores.mean():.4f}")
print(f"Std  CV Accuracy : {cv_scores.std():.4f}")
print(f"\nQ37: Single Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"     Mean CV Accuracy    : {cv_scores.mean():.4f}")
print(f"     Difference          : {abs(cv_scores.mean() - accuracy_score(y_test, y_pred_rf)):.4f}")
# Q39 — Summary table
models_info = {
    'Logistic Regression': (pipe_lr,   y_pred_lr,   y_proba_lr),
    'KNN':                 (pipe_knn,  y_pred_knn,  y_proba_knn),
    'SVM':                 (pipe_svm,  y_pred_svm,  y_proba_svm),
    'Decision Tree':       (pipe_tree, y_pred_tree, y_proba_tree),
    'Random Forest':       (pipe_rf,   y_pred_rf,   y_proba_rf),
}

best_params = {
    'Logistic Regression': f'C={best_C}',
    'KNN':                 f'k={best_k}',
    'SVM':                 f'kernel={best_svm["kernel"]}, C={best_svm["C"]}',
    'Decision Tree':       f'max_depth={best_max_d}, min_split={best_min_split}',
    'Random Forest':       f'n_est={best_n_est}, max_depth={best_rf_depth}, feat={best_rf_feat}',
}

rows = []
for name, (pipe, y_pred, y_proba) in models_info.items():
    _, _, thresholds = roc_curve(y_test, y_proba)
    fpr_, tpr_, _ = roc_curve(y_test, y_proba)
    rows.append({
        'Model': name,
        'Best Hyperparameters': best_params[name],
        'Test Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Test Precision': round(precision_score(y_test, y_pred), 4),
        'Test Recall':    round(recall_score(y_test, y_pred), 4),
        'Test F1':        round(f1_score(y_test, y_pred), 4),
        'Test ROC AUC':   round(auc(fpr_, tpr_), 4),
    })

summary_df = pd.DataFrame(rows).set_index('Model')
print("=== Q39: Model Comparison Summary ===")
print(summary_df.to_string())
# Visual comparison — grouped bar chart
metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test ROC AUC']
x = np.arange(len(metrics))
width = 0.15
colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']

fig, ax = plt.subplots(figsize=(14, 6))
for i, (model_name, row) in enumerate(summary_df.iterrows()):
    vals = [row[m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=model_name, color=colors[i], alpha=0.85, edgecolor='black')

ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics, rotation=15)
ax.set_ylabel('Score')
ax.set_ylim(0.5, 1.0)
ax.set_title('Q39: Model Comparison — All Test Metrics', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

best_f1  = summary_df['Test F1'].idxmax()
best_auc = summary_df['Test ROC AUC'].idxmax()
print(f"\nQ40: Highest Test F1-Score:  {best_f1} ({summary_df.loc[best_f1, 'Test F1']})")
print(f"     Highest Test ROC AUC:   {best_auc} ({summary_df.loc[best_auc, 'Test ROC AUC']})")
# Combined ROC curves — all 5 models
plt.figure(figsize=(8, 6))
roc_data = [
    ('Logistic Regression', fpr_lr,   tpr_lr,   auc_lr,   'steelblue'),
    ('KNN',                 fpr_knn,  tpr_knn,  auc_knn,  'darkorange'),
    ('SVM',                 fpr_svm,  tpr_svm,  auc_svm,  'green'),
    ('Decision Tree',       fpr_tree, tpr_tree, auc_tree, 'red'),
    ('Random Forest',       fpr_rf,   tpr_rf,   auc_rf,   'purple'),
]
for name, fpr, tpr, roc_auc, color in roc_data:
    plt.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All 5 Models', fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
