# Startup Exit Prediction: Acquired vs Closed

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


