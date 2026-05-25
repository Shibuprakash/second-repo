# =============================================================================
# Heart Disease Prediction - Lab Assessment
# Objectives:
#   1. Clean and prepare data
#   2. Build Random Forest with default hyperparameters
#   3. Tune hyperparameters using GridSearchCV
#   4. Compare with Logistic Regression (different regularization & penalties)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves to file instead of opening a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: LOAD AND EXPLORE DATA
# =============================================================================

# Load the CSV file into a pandas DataFrame
# Each row = one patient, each column = one medical feature
df = pd.read_csv("heart.csv")

print("=" * 60)
print("SECTION 1: DATA EXPLORATION")
print("=" * 60)

print("\n--- First 5 rows ---")
print(df.head())
# head() shows the first 5 rows so we can visually verify the data loaded correctly

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
# shape tells us how many patients (rows) and features (columns) we have

print("\n--- Column Data Types ---")
print(df.dtypes)
# dtypes tells us whether each column is numeric (int64/float64) or text (object)

print("\n--- Summary Statistics ---")
print(df.describe())
# describe() gives count, mean, std, min, max for each column
# Use this to spot outliers (e.g., unrealistically high cholesterol)

print("\n--- Missing Values per Column ---")
print(df.isnull().sum())
# isnull().sum() counts how many NaN (missing) values are in each column
# If any column has missing values, we need to handle them (fill or drop)

print("\n--- Target Variable Distribution ---")
print(df['output'].value_counts())
# output=1 means heart disease present, output=0 means absent
# We check if the dataset is balanced (roughly equal 0s and 1s)

# =============================================================================
# SECTION 2: DATA CLEANING AND PREPARATION
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: DATA CLEANING AND PREPARATION")
print("=" * 60)

# --- Step 2a: Drop Duplicate Rows ---
before = len(df)
df = df.drop_duplicates()
# drop_duplicates() removes any rows that are exact copies of another row
# Duplicate rows can bias the model by over-representing certain patients
after = len(df)
print(f"\nDuplicates removed: {before - after} rows (kept {after} rows)")

# --- Step 2b: Separate Features (X) and Target (y) ---
X = df.drop('output', axis=1)
# X = all columns EXCEPT 'output' — these are the input features (age, sex, bp, etc.)
# axis=1 means we are dropping a column (axis=0 would drop a row)

y = df['output']
# y = only the 'output' column — this is what we are trying to predict (0 or 1)

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target  (y) shape: {y.shape}")

# --- Step 2c: Feature Names for Reference ---
feature_names = X.columns.tolist()
print(f"\nFeature names: {feature_names}")

# --- Step 2d: Split into Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% of data goes to test set, 80% to training
    random_state=42,    # fixed seed so results are reproducible
    stratify=y          # ensures both train and test have same ratio of 0s and 1s
)
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# --- Step 2e: Feature Scaling (Standardization) ---
# Scaling is CRITICAL for Logistic Regression (but not needed for Random Forest)
# We still create scaled versions for use later in the comparison section

scaler = StandardScaler()
# StandardScaler transforms each feature so that:
#   mean = 0  and  standard deviation = 1
# This prevents features with large ranges (e.g., cholesterol 100-600)
# from dominating features with small ranges (e.g., fbs which is 0 or 1)

X_train_scaled = scaler.fit_transform(X_train)
# fit_transform on TRAINING data: learns the mean & std FROM training data, then scales it
# IMPORTANT: we fit ONLY on training data to prevent data leakage

X_test_scaled = scaler.transform(X_test)
# transform on TEST data: uses the SAME mean & std learned from training data
# We do NOT fit again — that would be data leakage (using test info during training)

print("\nFeature scaling applied (StandardScaler)")
print("  -> Used for Logistic Regression (Unscaled for Random Forest)")

# =============================================================================
# SECTION 3: RANDOM FOREST WITH DEFAULT HYPERPARAMETERS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: RANDOM FOREST — DEFAULT HYPERPARAMETERS")
print("=" * 60)

# Create a Random Forest Classifier with all default settings
rf_default = RandomForestClassifier(random_state=42)
# Random Forest builds many Decision Trees and combines their votes
# random_state=42 ensures the same trees are built every time we run the code

# --- What are the default hyperparameters? ---
print("\nDefault Hyperparameters:")
params = rf_default.get_params()
for key, val in params.items():
    print(f"  {key:25s} = {val}")
# get_params() shows all hyperparameters and their current (default) values

# Train the model on training data
rf_default.fit(X_train, y_train)
# fit() = training phase: the model learns patterns from X_train and y_train
# It builds 100 decision trees (n_estimators=100 by default) on random subsets

# Predict on test data
y_pred_rf_default = rf_default.predict(X_test)
# predict() = the 100 trees each vote, and the majority vote becomes the prediction

# Evaluate the default model
acc_rf_default = accuracy_score(y_test, y_pred_rf_default)
auc_rf_default = roc_auc_score(y_test, rf_default.predict_proba(X_test)[:, 1])
# accuracy_score: fraction of predictions that are correct
# roc_auc_score: measures how well the model separates class 0 from class 1
#   AUC = 1.0 is perfect, 0.5 is random guessing

print(f"\nDefault RF — Accuracy : {acc_rf_default:.4f}")
print(f"Default RF — ROC-AUC  : {auc_rf_default:.4f}")

print("\nClassification Report (Default RF):")
print(classification_report(y_test, y_pred_rf_default, target_names=['No Disease', 'Disease']))
# Shows Precision, Recall, F1-score for each class
# Precision: of all predicted "Disease", how many actually have it?
# Recall   : of all actual "Disease" patients, how many did we catch?
# F1-score : harmonic mean of Precision and Recall

# --- Feature Importance ---
print("\nTop 10 Feature Importances (Default RF):")
importances = rf_default.feature_importances_
# feature_importances_ tells us which features the trees relied on the most
# Higher value = more important feature for prediction

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feat_imp_df.to_string(index=False))

# =============================================================================
# SECTION 4: HYPERPARAMETER TUNING WITH GRID SEARCH CV
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: HYPERPARAMETER TUNING — GRID SEARCH CV")
print("=" * 60)

# --- Understanding the Hyperparameters ---
print("""
KEY RANDOM FOREST HYPERPARAMETERS:

1. n_estimators   : Number of trees in the forest.
                    More trees -> more stable predictions, but slower training.
                    Default = 100

2. max_depth      : Maximum depth each tree can grow.
                    None = unlimited (trees grow until leaves are pure).
                    Deeper trees = more complex, may overfit.

3. min_samples_split : Minimum samples required to split an internal node.
                       Higher value -> simpler trees, less overfitting.

4. min_samples_leaf  : Minimum samples required in each leaf node.
                       Higher value -> smoother model, avoids fitting noise.

5. max_features   : Number of features considered at each split.
                    'sqrt' = sqrt(total features) — default for classification.
                    Randomness here is what makes each tree different.

6. criterion      : How to measure split quality.
                    'gini' (Gini Impurity) or 'entropy' (Information Gain).
""")

# Define the hyperparameter grid to search over
param_grid_rf = {
    'n_estimators'    : [50, 100, 200],
    # Try forests with 50, 100, or 200 trees

    'max_depth'       : [None, 5, 10],
    # None = unlimited depth; 5 and 10 cap the tree depth to prevent overfitting

    'min_samples_split': [2, 5, 10],
    # Minimum samples needed before splitting a node

    'min_samples_leaf' : [1, 2, 4],
    # Minimum samples that must remain in a leaf after a split

    'max_features'    : ['sqrt', 'log2'],
    # 'sqrt' = square root of features; 'log2' = log base 2 of features
}

# GridSearchCV exhaustively tries every combination of the above
# Total combinations = 3 × 3 × 3 × 3 × 2 = 162 combinations
# With cv=5 (5-fold cross-validation), it trains 162 × 5 = 810 models

print("\nStarting Grid Search (this may take a minute)...")
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=5,               # 5-fold cross-validation: splits training data into 5 parts,
                        # trains on 4, validates on 1, rotates 5 times
    scoring='accuracy', # metric used to pick the best hyperparameter combo
    n_jobs=-1,          # use all available CPU cores for parallel processing
    verbose=1           # print progress
)

grid_search_rf.fit(X_train, y_train)
# fit() runs all 810 training/validation cycles and records the accuracy of each

print("\nBest Hyperparameters found by Grid Search:")
print(grid_search_rf.best_params_)
# best_params_ gives the hyperparameter combination with the highest CV accuracy

print(f"\nBest Cross-Validation Accuracy: {grid_search_rf.best_score_:.4f}")
# best_score_ is the average accuracy across all 5 folds for the best model

# Evaluate the best tuned model on the test set
best_rf = grid_search_rf.best_estimator_
# best_estimator_ is the already-fitted model with the best hyperparameters

y_pred_rf_tuned = best_rf.predict(X_test)
acc_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
auc_rf_tuned  = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print(f"\nTuned RF — Accuracy : {acc_rf_tuned:.4f}")
print(f"Tuned RF — ROC-AUC  : {auc_rf_tuned:.4f}")

print("\nClassification Report (Tuned RF):")
print(classification_report(y_test, y_pred_rf_tuned, target_names=['No Disease', 'Disease']))

# =============================================================================
# SECTION 5: LOGISTIC REGRESSION — COMPARATIVE ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: LOGISTIC REGRESSION — COMPARATIVE ANALYSIS")
print("=" * 60)

print("""
LOGISTIC REGRESSION KEY CONCEPTS:

Penalty (Regularization Type):
  - 'l1' (Lasso)  : Adds |weights| to the loss function.
                    Drives some feature weights to exactly 0 -> sparse model.
                    Good for feature selection.
  - 'l2' (Ridge)  : Adds weights² to the loss function.
                    Shrinks all weights toward 0, but never exactly 0.
                    Default and most commonly used.
  - 'elasticnet'  : Combination of L1 and L2.
  - 'none'        : No regularization (risky — may overfit).

C (Inverse Regularization Strength):
  - C = 1/lambda  (where lambda is the regularization strength)
  - SMALL C  -> STRONG regularization -> simpler model -> may underfit
  - LARGE C  -> WEAK  regularization -> complex model -> may overfit
  - Default C = 1.0
""")

# We will try L1 and L2 penalties with multiple values of C
penalties    = ['l1', 'l2']
C_values     = [0.001, 0.01, 0.1, 1, 10, 100]
# Spanning a wide range of C lets us see the full effect of regularization strength

lr_results = []  # store results for comparison table

print(f"\n{'Penalty':<10} {'C':>8} {'Accuracy':>12} {'ROC-AUC':>12}")
print("-" * 45)

for penalty in penalties:
    for C in C_values:

        # Choose the solver compatible with the penalty
        # 'liblinear' supports both L1 and L2
        # 'lbfgs' only supports L2 (would fail for L1)
        solver = 'liblinear'

        lr = LogisticRegression(
            penalty=penalty,    # regularization type
            C=C,                # regularization strength (inverse)
            solver=solver,      # optimization algorithm
            max_iter=1000,      # maximum iterations for convergence
            random_state=42
        )

        lr.fit(X_train_scaled, y_train)
        # We use SCALED data for Logistic Regression
        # Scaling ensures all features contribute equally during optimization

        y_pred_lr = lr.predict(X_test_scaled)
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
        # predict_proba returns [prob_class0, prob_class1] for each sample
        # We take [:, 1] to get the probability of heart disease

        acc = accuracy_score(y_test, y_pred_lr)
        auc = roc_auc_score(y_test, y_prob_lr)

        lr_results.append({
            'Penalty': penalty,
            'C': C,
            'Accuracy': acc,
            'ROC-AUC': auc
        })

        print(f"{penalty:<10} {C:>8} {acc:>12.4f} {auc:>12.4f}")

# Convert results to DataFrame for easy analysis
lr_df = pd.DataFrame(lr_results)

print("\n--- Best Logistic Regression Configuration ---")
best_lr_row = lr_df.loc[lr_df['Accuracy'].idxmax()]
# idxmax() finds the index of the row with the highest accuracy
print(best_lr_row.to_string())

# =============================================================================
# SECTION 6: FINAL COMPARATIVE ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: FINAL COMPARATIVE ANALYSIS")
print("=" * 60)

# Retrain the best Logistic Regression for final comparison
best_penalty = best_lr_row['Penalty']
best_C       = best_lr_row['C']

best_lr_model = LogisticRegression(
    penalty=best_penalty,
    C=best_C,
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
best_lr_model.fit(X_train_scaled, y_train)
y_pred_best_lr = best_lr_model.predict(X_test_scaled)
auc_best_lr    = roc_auc_score(y_test, best_lr_model.predict_proba(X_test_scaled)[:, 1])
acc_best_lr    = accuracy_score(y_test, y_pred_best_lr)

# Summary comparison table
print("\n" + "=" * 55)
print(f"{'Model':<35} {'Accuracy':>10} {'ROC-AUC':>10}")
print("=" * 55)
print(f"{'Random Forest (Default)':<35} {acc_rf_default:>10.4f} {auc_rf_default:>10.4f}")
print(f"{'Random Forest (Tuned)':<35} {acc_rf_tuned:>10.4f} {auc_rf_tuned:>10.4f}")
print(f"{'Best Logistic Regression':<35} {acc_best_lr:>10.4f} {auc_best_lr:>10.4f}")
print(f"  (penalty={best_penalty}, C={best_C})")
print("=" * 55)

# --- Cross-Validation Scores for Fair Comparison ---
print("\n--- 5-Fold Cross-Validation Scores (on full dataset) ---")
cv_rf  = cross_val_score(RandomForestClassifier(**grid_search_rf.best_params_, random_state=42),
                          X, y, cv=5, scoring='accuracy')
cv_lr  = cross_val_score(LogisticRegression(penalty=best_penalty, C=best_C,
                          solver='liblinear', max_iter=1000, random_state=42),
                          scaler.fit_transform(X), y, cv=5, scoring='accuracy')

print(f"Tuned RF  — CV Accuracy: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
print(f"Best LR   — CV Accuracy: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")

# =============================================================================
# SECTION 7: VISUALIZATIONS
# =============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Heart Disease Prediction — Analysis Dashboard", fontsize=15, fontweight='bold')

# --- Plot 1: Target Distribution ---
ax = axes[0, 0]
df['output'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'], edgecolor='black')
ax.set_title("Target Distribution")
ax.set_xticklabels(['No Disease (0)', 'Disease (1)'], rotation=0)
ax.set_ylabel("Count")
# This shows whether the dataset is balanced

# --- Plot 2: Feature Importances (Tuned RF) ---
ax = axes[0, 1]
feat_imp_tuned = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=True)
feat_imp_tuned.plot(kind='barh', ax=ax, color='teal', edgecolor='black')
ax.set_title("Feature Importances (Tuned RF)")
ax.set_xlabel("Importance Score")
# Higher bar = feature has more influence on prediction

# --- Plot 3: Confusion Matrix (Tuned RF) ---
ax = axes[0, 2]
cm_rf = confusion_matrix(y_test, y_pred_rf_tuned)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
ax.set_title("Confusion Matrix (Tuned RF)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
# TN (top-left) : correctly predicted No Disease
# TP (bottom-right): correctly predicted Disease
# FP (top-right) : No Disease predicted as Disease (false alarm)
# FN (bottom-left): Disease predicted as No Disease (dangerous miss)

# --- Plot 4: LR Accuracy vs C for L1 and L2 ---
ax = axes[1, 0]
for penalty in penalties:
    subset = lr_df[lr_df['Penalty'] == penalty]
    ax.plot(subset['C'], subset['Accuracy'], marker='o', label=f'{penalty.upper()} Penalty')
ax.set_xscale('log')
# log scale because C spans 0.001 to 100 — a very wide range
ax.set_title("LR Accuracy vs C (Regularization Strength)")
ax.set_xlabel("C (log scale)   ->   Larger C = Weaker Regularization")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.4)

# --- Plot 5: LR ROC-AUC vs C ---
ax = axes[1, 1]
for penalty in penalties:
    subset = lr_df[lr_df['Penalty'] == penalty]
    ax.plot(subset['C'], subset['ROC-AUC'], marker='s', label=f'{penalty.upper()} Penalty')
ax.set_xscale('log')
ax.set_title("LR ROC-AUC vs C")
ax.set_xlabel("C (log scale)")
ax.set_ylabel("ROC-AUC")
ax.legend()
ax.grid(True, alpha=0.4)

# --- Plot 6: Final Model Comparison Bar Chart ---
ax = axes[1, 2]
model_names  = ['RF Default', 'RF Tuned', f'Best LR\n(pen={best_penalty}, C={best_C})']
accuracies   = [acc_rf_default, acc_rf_tuned, acc_best_lr]
auc_scores   = [auc_rf_default, auc_rf_tuned, auc_best_lr]

x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', edgecolor='black')
ax.bar(x + width/2, auc_scores,  width, label='ROC-AUC',  color='tomato',    edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylim(0.7, 1.05)
ax.set_title("Final Model Comparison")
ax.set_ylabel("Score")
ax.legend()
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig("heart_disease_analysis.png", dpi=150, bbox_inches='tight')
print("Visualization saved as 'heart_disease_analysis.png'")
plt.close()

print("\n" + "=" * 60)
print("LAB ASSESSMENT COMPLETE")
print("=" * 60)
