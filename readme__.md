# Startup Exit Prediction: Acquired vs Closed

Q. Which startup characteristics are most strongly associated with successful acquisition?
Ans : The logistic regression model identified Market Sector as the most influential group of characteristics, indicating that the industry in which a startup operates is strongly associated with whether it is ultimately acquired or closed. other characteristics are higher funding , longer funding duration and venture are positively associated with acquisition. 

Q. Which funding type has the strongest association with acquisition?
Ans : Among all funding mechanism, venture is the most strongest association with acquition. It had the largest relative contribution among funding-type variables (2.72%) and a negative logistic regression coefficient, indicating that venture-backed startups had lower odds of closure and therefore higher odds of acquisition after accounting for funding amount, market sector, fundraising duration, and other funding sources.

Q. Does market sector matter?
Ans : Yes. The analysis indicates that the startup's market sector is one of the strongest characteristics associated with whether a startup is acquired or closed. The logistic regression model consistently assigned substantial weight to market sector, suggesting that industry choice plays an important role in startup outcomes.

More strongly associated with acquisition market sectors: 
1. AI & Data
2. Security
3. Financial Services
4. Travel & Hospitality
5. Education 


## Project Overview

This project aims to identify the factors that differentiate startups that were eventually **acquired** from those that were **closed**. The objective is to build an analytics-driven machine learning solution that combines exploratory data analysis, statistical analysis, feature engineering, predictive modeling, and explainable AI.

Unlike many startup success prediction projects, this project focuses only on **terminal business outcomes** (Acquired vs Closed), making the classification problem well-defined and suitable for supervised learning.

---

# Data Cleaning

The original Crunchbase investment dataset required several preprocessing steps before exploratory analysis and model development.

## 1. Column Name Standardization

Leading and trailing whitespaces were removed from all column names to ensure consistent column referencing throughout the project.

```python
df.columns = df.columns.str.strip()
```

---

## 2. Initial Data Inspection

The dataset structure was examined using:

* Dataset shape
* Data types
* Summary information
* Initial records

This helped identify missing values, incorrect data types, and potential data quality issues.

---

## 3. Duplicate Removal

Duplicate observations were checked and removed to avoid redundant information during analysis and model training.

---

## 4. Removal of Non-Informative Columns

The following columns were removed:

| Column       | Reason                                                              |
| ------------ | ------------------------------------------------------------------- |
| permalink    | Unique company identifier                                           |
| name         | Company name has extremely high cardinality and no predictive value |
| homepage_url | Unique website URL with no analytical significance                  |

These variables do not contribute meaningful information for predictive modeling and may introduce unnecessary noise.

---

## 5. Missing Value Assessment

Missing values were quantified for every feature.

At this stage, missing values were **documented but not immediately imputed**, since the appropriate treatment depends on the exploratory analysis and feature engineering phase.

---

## 6. Target Variable Definition

The original dataset contained four categories:

* acquired
* closed
* operating
* missing (NaN)

For this project, only startups with known terminal outcomes were retained.

### Included

* acquired
* closed

### Excluded

* operating
* missing status

Reasons:

* Operating companies have not yet reached a final business outcome.
* Missing status records cannot be used for supervised learning because the target label is unavailable.

A separate copy of the missing-status observations was preserved for potential future inference.

---

## 7. Data Type Conversion

The following columns were converted into datetime format:

* founded_at
* first_funding_at
* last_funding_at

The `funding_total_usd` column was cleaned and converted into a numeric data type after removing formatting inconsistencies.

---

## 8. Data Quality Validation

Several logical consistency checks were performed.

### Funding Amount

* Negative funding amounts

### Funding Rounds

* Negative funding rounds

### Temporal Consistency

The following temporal relationships were validated:

* First funding date before company founding date
* Last funding date before first funding date
* Invalid future dates

---

## 9. Investigation of Temporal Anomalies

A total of **447 companies** were identified where the recorded first funding date occurred before the recorded founding date.

Rather than removing all such observations, each anomaly was investigated.

Most companies exhibited small differences (typically a few months), which are plausible because startups often receive angel or seed funding before formal legal incorporation.

Only records satisfying **all** of the following conditions were treated as true anomalies:

* Single funding round
* First funding date equals last funding date
* Funding occurred more than five years before the recorded founding date

These records were considered internally inconsistent and removed from the dataset.

This evidence-based approach preserves realistic startup funding behavior while eliminating obvious data quality errors.

---

# Current Project Status

✔ Data loading

✔ Column name standardization

✔ Duplicate inspection

✔ Removal of identifier columns

✔ Missing value assessment

✔ Target variable definition

✔ Data type conversion

✔ Funding amount cleaning

✔ Data quality validation

✔ Temporal anomaly detection

✔ Removal of confirmed anomalous records

---

# Business Questions

Q. Do startup typically raise multiple funding rounds, or do most rely on only one or two rounds of funding??
The distribution is highly right skewed. which indicates that most of the startsup raise small rounds of funds , where small numbers of startups raise high rounds of funds. The median of funding round is 1, but median is 1.77 which means distribution is influenced by outfliers.
Approximately 60% of the startups are raised funding only once and 81% of the startups are raised funding twice. Only 0.1% of the startsups are raised funding 10 or more rounds. 

# Key Business Insights from Exploratory Data Analysis

*Acquired startups consistently raise substantially more funding than startups that eventually close. This suggests that access to capital is strongly associated with successful startup outcomes.

* Startup funding is not distributed evenly. A small number of companies attract exceptionally large investments, while the majority raise relatively modest amounts.

* Startups that progress through multiple funding rounds are acquired more frequently than those that stop after one or two rounds, indicating that sustained fundraising is associated with better long-term outcomes.

* Funding characteristics emerge as one of the most informative aspects of startup performance, highlighting the importance of a company's fundraising journey when evaluating its future outcome.


# Feature Engineering: Market Sector Consolidation

Objective

Reduce the high cardinality of the market feature while preserving meaningful business information.

Approach

Consolidated more than 350 market categories into 17 business sectors using domain knowledge.
Grouped semantically related categories (e.g., Software, SaaS, Cloud Computing → Software & Cloud).
Assigned infrequent or miscellaneous categories to Other.

Outcome

Original unique market categories: 350+
Final business sectors: 17 + Other
Other category: 8.44% of observations

That demonstrates both the motivation and the result of your engineering decision.

# Before Hypertuning

Built three models: Logistic Regression, Random Forest, and XGBoost.
Compared them using Accuracy, Precision, Recall, F1, and ROC-AUC.
Discarded Random Forest because it consistently underperformed.
Compared Logistic Regression and XGBoost.
Selected Logistic Regression as the final model because:
It achieved the highest ROC-AUC (0.7366) and highest accuracy (69.66%).
Its performance was essentially tied with XGBoost.
It provides clear coefficient-based explanations, making it a better fit for an analytics project focused on understanding what distinguishes acquired startups from closed ones.

Three classification models—Logistic Regression, Random Forest, and XGBoost—were evaluated using a stratified 5-fold cross-validation strategy. Logistic Regression achieved the highest mean ROC-AUC (0.7396) and the highest accuracy (69.37%), while maintaining competitive performance across all evaluation metrics. Although XGBoost achieved a higher F1-score by improving recall, Logistic Regression provided superior overall discrimination, greater stability across folds, and substantially better interpretability. Given the project's objective of understanding the factors that distinguish acquired startups from closed startups, Logistic Regression was selected as the final model.

Hyperparameter tuning provided a modest but consistent improvement across all evaluation metrics. The tuned Logistic Regression model achieved the highest overall performance and was selected as the final model.