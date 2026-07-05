# Methodology: Statistical and Machine Learning Methods

**Project:** Startup Exit Prediction — Acquired vs. Closed  
**Dataset:** Crunchbase VC Investment Dataset (54,294 records, 39 features)  
**Objective:** Identify characteristics that distinguish startups that get acquired from those that shut down, using a binary classification framework.

---

## Overview

This document describes every statistical and machine learning method applied in this project. For each method, it explains where in the pipeline it was used and the specific reason it was chosen over alternatives.

The pipeline is organized into five stages:

1. Data Cleaning and Validation
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing
5. Model Training, Evaluation, and Interpretation

---

## Stage 1: Data Cleaning and Validation

### 1.1 Descriptive Statistics for Initial Inspection

**Where used:** `Scripts/data_cleaning.ipynb` — initial data inspection  
**What:** Summary statistics (shape, dtypes, null counts, value counts) computed using `df.info()`, `df.describe()`, and `df.value_counts()`.  
**Why:** Before any transformation, it is necessary to understand the structure of the dataset — how many records exist, what data types are present, and where missing values are concentrated. These checks expose obvious quality problems early, preventing silent errors in downstream steps.

---

### 1.2 Duplicate Detection

**Where used:** `Scripts/data_cleaning.ipynb`  
**What:** Identification and removal of exact duplicate rows using `df.duplicated()`.  
**Why:** Duplicate records inflate dataset size without adding new information. In a classification task, they can cause a model to effectively memorize repeated observations, leading to overfitting and artificially inflated training accuracy. Removing them ensures that each company contributes exactly one observation.

---

### 1.3 Logical Consistency Checks (Rule-Based Validation)

**Where used:** `Scripts/data_cleaning.ipynb`  
**What:** Custom validation rules applied to detect:
- Negative funding amounts
- Negative funding round counts
- First funding date occurring before the company's founding date
- Last funding date occurring before the first funding date
- Invalid future-dated records

**Why:** Domain knowledge dictates what values are physically possible. Negative funding and rounds are impossible by definition. Temporal ordering violations indicate data entry errors. These checks are a form of constraint-based data validation — a standard practice in data engineering that prevents logically impossible records from distorting statistical summaries and model training.

---

### 1.4 Evidence-Based Anomaly Treatment (Threshold Filtering)

**Where used:** `Scripts/data_cleaning.ipynb` — temporal anomaly investigation  
**What:** Among the 447 companies where first funding predated founding, records were only removed if they simultaneously satisfied three conditions: (1) single funding round, (2) first funding date equals last funding date, and (3) funding gap exceeds five years.  
**Why:** Blanket deletion of all temporal anomalies would have discarded valid records, since early-stage funding before legal incorporation is a documented real-world phenomenon. A rule-based threshold was used to distinguish plausible from implausible observations, preserving data volume while removing only confirmed errors. This conservative approach reduces type I errors in the cleaning step.

---

## Stage 2: Exploratory Data Analysis

### 2.1 Univariate Distribution Analysis

**Where used:** `Scripts/data_eda.ipynb`  
**What:** Frequency distributions, histograms, and summary statistics (mean, median, percentiles) computed for each feature individually.  
**Why:** Univariate analysis reveals the shape of each variable's distribution independently of any other variable. For funding amounts and funding rounds, this exposed severe right skew — a small number of companies raised disproportionately large amounts. Understanding this skew informed decisions about which transformations to apply in preprocessing and which statistical measures (median rather than mean) are more representative.

---

### 2.2 Skewness Analysis

**Where used:** `Scripts/data_eda.ipynb`  
**What:** Assessment of distributional asymmetry for continuous variables, particularly `funding_total_usd` and `funding_rounds`.  
**Why:** The funding distribution was found to be highly right-skewed: the median number of funding rounds was 1, but the mean was approximately 1.77, pulled upward by a small number of high-round companies. Recognizing skewness is important because it affects which statistical summaries are reliable and whether transformations (such as log scaling) are needed before modeling.

---

### 2.3 Bivariate Comparative Analysis (Group-Wise Statistics)

**Where used:** `Scripts/data_eda.ipynb`  
**What:** Comparison of feature distributions between acquired and closed groups using grouped summary statistics and box plots.  
**Why:** The core business question is about *differences between groups*. Bivariate analysis directly answers whether acquired startups look systematically different from closed ones across funding, sector, and timing dimensions. This comparison forms the analytical basis for feature selection — variables that show clear group separation are likely to have predictive value.

---

### 2.4 Frequency Analysis and Cumulative Distribution

**Where used:** `Scripts/data_eda.ipynb`  
**What:** Percentile-based analysis to compute cumulative proportions (e.g., what percentage of startups have only one funding round, what percentage have two or fewer).  
**Why:** Cumulative distributions provide a more complete picture of a variable's spread than simple averages. The finding that 60% of startups raised exactly one round and 81% raised at most two rounds was derived from this approach, illustrating the concentrated nature of the funding landscape and highlighting that multi-round companies are a distinct minority.

---

### 2.5 Categorical Frequency and Cross-Tabulation

**Where used:** `Scripts/data_eda.ipynb`  
**What:** Value counts and cross-tabulation of categorical variables (market category, country, funding type) against the target variable (acquired vs. closed).  
**Why:** For categorical predictors, frequency tables and cross-tabulations reveal which categories are more prevalent in each outcome class. This analysis guided the market sector consolidation step by identifying which categories were too sparse to model independently and which showed distinctive outcome patterns.

---

## Stage 3: Feature Engineering

### 3.1 Cardinality Reduction via Domain-Knowledge Consolidation

**Where used:** `Scripts/feature_Engineering.ipynb`  
**What:** The `market` column, which originally contained over 350 distinct categories, was manually consolidated into 17 business sectors plus an "Other" category using domain expertise. Semantically related categories (e.g., Software, SaaS, Cloud Computing, Enterprise Software) were grouped into a single sector (Software & Cloud).  
**Why:** High cardinality categorical variables present two problems for machine learning models. First, one-hot encoding 350+ categories would produce an extremely sparse feature matrix, making it difficult for models to learn stable patterns. Second, many individual categories have too few observations to produce reliable estimates. Consolidation into 17 sectors reduces noise, improves statistical power per category, and makes the resulting feature space interpretable. The "Other" category captures 8.44% of observations that do not clearly belong to any defined sector.

---

### 3.2 Temporal Feature Derivation

**Where used:** `Scripts/feature_Engineering.ipynb`  
**What:** Derived features were computed from date columns, including fundraising duration (time between first and last funding date).  
**Why:** Raw date columns (first funding date, last funding date, founding date) are not directly usable as model features. Derived numeric quantities — such as how long a company has been actively fundraising — encode meaningful business information that the raw dates do not express on their own. Fundraising duration, for example, captures whether a startup sustained investor interest over time, which is a separate signal from total funding amount alone.

---

## Stage 4: Data Preprocessing

### 4.1 One-Hot Encoding

**Where used:** `Scripts/data_preprocessing.ipynb`  
**What:** Nominal categorical variables (market sector, country code) were transformed into binary indicator columns using one-hot encoding.  
**Why:** Machine learning models operate on numeric inputs. One-hot encoding is the standard transformation for nominal categorical variables because it makes no assumption about ordering or magnitude between categories — unlike integer label encoding, which would incorrectly imply that "sector 3" is greater than "sector 2". After cardinality reduction in feature engineering, the resulting number of one-hot columns was manageable.

---

### 4.2 Standard Scaling (Z-Score Normalization)

**Where used:** `Scripts/data_preprocessing.ipynb`  
**What:** Continuous numeric features (funding amount, funding rounds, fundraising duration) were scaled to zero mean and unit variance using `StandardScaler`.  
**Why:** Logistic Regression uses gradient-based optimization, and its convergence behavior and coefficient magnitudes are sensitive to the scale of input features. Without scaling, a feature measured in millions of dollars would numerically dominate a feature measured as a binary 0/1 indicator, leading to poorly calibrated coefficients and slower convergence. Standard scaling places all features on a comparable numeric scale, ensuring that the model treats each feature's variance equally during optimization. Note: tree-based models (Random Forest, XGBoost) are scale-invariant, but scaling was applied uniformly for consistency across model comparisons.

---

### 4.3 Stratified Train-Test Split

**Where used:** `Scripts/data_preprocessing.ipynb`  
**What:** The labeled dataset (6,295 records) was divided into training and test sets using stratified sampling, preserving the original class ratio (approximately 59% acquired, 41% closed) in both subsets.  
**Why:** The dataset has a class imbalance — acquired companies outnumber closed companies. Without stratification, random splitting could produce a test set with a different class distribution than the training set, making evaluation metrics unreliable. Stratified splitting guarantees that both subsets reflect the true population ratio, producing evaluation metrics that generalize to new data.

---

## Stage 5: Model Training, Evaluation, and Interpretation

### 5.1 Stratified K-Fold Cross-Validation (k = 5)

**Where used:** `main.ipynb` — model evaluation  
**What:** Model performance was estimated using 5-fold cross-validation where the data was split into 5 folds, with each fold serving as the validation set once. The class ratio was preserved in every fold via stratification.  
**Why:** With only 6,295 labeled examples, a single train-test split produces performance estimates that are sensitive to which specific records happen to fall in the test set. Cross-validation reduces this variance by averaging performance across five different splits, producing a more stable and reliable estimate of generalization performance. Stratification within each fold ensures the class imbalance does not distort any individual fold's evaluation.

---

### 5.2 Logistic Regression

**Where used:** `main.ipynb` — primary classification model  
**What:** A linear probabilistic classifier that models the log-odds of the binary outcome (acquired = 1, closed = 0) as a linear combination of input features. The model was trained with L2 regularization (Ridge penalty).  
**Why (choice rationale):**
- **Interpretability:** Each coefficient directly quantifies the direction and relative magnitude of a feature's association with the outcome. This is essential for answering the project's research questions (which sector, which funding type, which characteristics matter most).
- **Calibrated probabilities:** Logistic Regression outputs well-calibrated probability estimates, making the model's predictions interpretable as genuine likelihoods.
- **L2 regularization:** With many binary one-hot features, regularization prevents overfitting by shrinking coefficients of less informative features toward zero.
- **Performance:** Achieved the highest ROC-AUC (0.7366) and accuracy (69.66%) among all three candidate models.

---

### 5.3 Random Forest

**Where used:** `main.ipynb` — candidate model (eliminated)  
**What:** An ensemble of decision trees, each trained on a bootstrap sample of the data with a random subset of features considered at each split. Predictions are made by majority vote across all trees.  
**Why (choice rationale):**
- Random Forest was evaluated as a strong baseline for tabular classification tasks due to its robustness to outliers, ability to model non-linear interactions, and implicit feature selection.
- **Why it was eliminated:** It achieved the lowest ROC-AUC (0.7071) and lowest accuracy (67.59%) among the three models, indicating that the non-linear interactions it captures do not provide additional predictive value in this dataset. The relatively small labeled dataset (6,295 records) may also limit the diversity benefit of the ensemble.

---

### 5.4 XGBoost (Extreme Gradient Boosting)

**Where used:** `main.ipynb` — candidate model  
**What:** A gradient boosting framework that builds an ensemble of shallow decision trees sequentially, where each tree corrects the residual errors of the previous ensemble. Uses second-order gradient information and regularization terms for improved convergence and control over overfitting.  
**Why (choice rationale):**
- XGBoost is one of the most effective algorithms for structured/tabular classification tasks and was evaluated for its ability to capture complex non-linear relationships and feature interactions.
- It achieved higher recall (0.6000) and F1-score (0.6160) than Logistic Regression by identifying more true acquisitions.
- **Why it was not selected as the final model:** Its ROC-AUC (0.7325) and accuracy (69.10%) were lower than Logistic Regression, and it provides no direct coefficient-based explanation — making it less suitable for an analytics project where understanding the *drivers* of acquisition is as important as prediction accuracy.

---

### 5.5 Hyperparameter Tuning

**Where used:** `main.ipynb` — applied to Logistic Regression  
**What:** Systematic search over regularization strength (the `C` parameter) and solver settings for Logistic Regression.  
**Why:** The default hyperparameters of any model are not necessarily optimal for a specific dataset. Tuning the regularization strength controls the bias-variance trade-off — a smaller C applies stronger regularization and may underfit, while a larger C applies weaker regularization and may overfit. Tuning identified a configuration that improved ROC-AUC from 0.7366 to 0.7396 and overall accuracy to 69.37%.

---

### 5.6 Evaluation Metrics

**Where used:** `main.ipynb` — model comparison  
**What:** Five metrics were computed for each model:

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Accuracy | (TP + TN) / Total | Overall correct classification rate |
| Precision | TP / (TP + FP) | Of predicted acquisitions, how many were truly acquired |
| Recall | TP / (TP + FN) | Of all truly acquired startups, how many were identified |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under the ROC curve | Probability that the model ranks a randomly chosen acquired startup above a randomly chosen closed startup |

**Why this set of metrics:**
- **Accuracy** alone is insufficient with class imbalance — a model predicting all startups as acquired would achieve 58.6% accuracy trivially.
- **Precision and Recall** capture the trade-off between false positives (predicting acquisition when the company actually closed) and false negatives (missing true acquisitions). Depending on the use case, either direction of error may matter more.
- **F1 Score** summarizes the Precision-Recall trade-off into a single number, useful for comparing models where both error types matter.
- **ROC-AUC** is the primary selection criterion because it measures discriminative ability across all possible classification thresholds, making it threshold-agnostic and robust to class imbalance. It answers the question: how well can the model rank acquired startups above closed ones?

---

### 5.7 Coefficient-Based Feature Importance (Logistic Regression Explainability)

**Where used:** `main.ipynb` — model interpretation  
**What:** The magnitude and sign of each logistic regression coefficient were examined to determine which features had the strongest association with acquisition vs. closure.  
**Why:** In Logistic Regression, each coefficient represents the change in the log-odds of the positive class (acquisition) for a one-unit increase in that feature, holding all other features constant. Larger absolute coefficient values indicate stronger association. This approach directly answers the project's research questions without needing additional explainability tools — the model is inherently interpretable, and its coefficients provide actionable, quantified insights about each predictor's contribution.

Key findings from coefficient analysis:
- Market sector variables collectively had the largest total contribution.
- Venture capital funding carried the largest individual coefficient among funding-type variables (relative contribution: 2.72%).
- Higher total funding and longer fundraising duration were positively associated with acquisition (negative coefficients for closure).

---

## Summary Table

| Method | Stage | Category | Reason for Use |
|--------|-------|----------|---------------|
| Descriptive Statistics | Data Cleaning | Statistical | Initial data profiling and quality assessment |
| Duplicate Detection | Data Cleaning | Statistical | Remove redundant observations |
| Rule-Based Validation | Data Cleaning | Statistical | Enforce domain constraints on values |
| Threshold-Based Anomaly Filtering | Data Cleaning | Statistical | Distinguish plausible from implausible records |
| Univariate Distribution Analysis | EDA | Statistical | Understand variable distributions and detect skew |
| Skewness Analysis | EDA | Statistical | Quantify asymmetry in funding distributions |
| Bivariate Group Comparison | EDA | Statistical | Identify features that differ between outcome groups |
| Cumulative Frequency Analysis | EDA | Statistical | Characterize concentration of funding activity |
| Categorical Cross-Tabulation | EDA | Statistical | Relate categorical features to target variable |
| Cardinality Reduction | Feature Engineering | Domain Knowledge | Reduce noise, improve statistical power per category |
| Temporal Feature Derivation | Feature Engineering | Feature Engineering | Extract business-meaningful signals from date fields |
| One-Hot Encoding | Preprocessing | Data Transformation | Convert nominal categories to numeric features |
| Standard Scaling | Preprocessing | Data Transformation | Normalize feature scales for gradient-based optimization |
| Stratified Train-Test Split | Preprocessing | Statistical | Preserve class ratio across data splits |
| Stratified K-Fold Cross-Validation | Model Evaluation | Statistical | Produce stable, low-variance performance estimates |
| Logistic Regression | Modeling | Machine Learning | Interpretable linear classifier; highest ROC-AUC |
| Random Forest | Modeling | Machine Learning | Evaluated; eliminated due to inferior performance |
| XGBoost | Modeling | Machine Learning | Evaluated; strong recall but lower overall discrimination |
| Hyperparameter Tuning | Modeling | Machine Learning | Optimize regularization to improve generalization |
| ROC-AUC, Accuracy, Precision, Recall, F1 | Evaluation | Statistical | Comprehensive, threshold-agnostic model comparison |
| Coefficient Analysis | Interpretation | Statistical / Explainability | Quantify feature-level association with acquisition |

---

*This document covers all statistical and machine learning methods applied in the Startup Exit Prediction project.*
