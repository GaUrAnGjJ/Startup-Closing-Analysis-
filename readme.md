# Startup Exit Prediction: Acquired vs. Closed

> *Can data tell us which startups survive and which ones disappear? This project set out to find the answer.*

---

## Overview

Every year, thousands of startups compete for funding, customers, and a place in the market. Most will eventually reach a crossroads — some get acquired by larger companies, validating years of hard work, while others quietly shut their doors. What separates these two outcomes?

This project explores that exact question. Using a real-world dataset of **54,294 venture-capital-backed startups** sourced from Crunchbase, we built an end-to-end machine learning pipeline that identifies the characteristics most strongly associated with a startup being **acquired** versus **closed**.

Unlike many startup analysis projects that try to predict "success" broadly, this project focuses only on **terminal business outcomes** — companies that have definitively reached the end of their journey, either through acquisition or closure. This makes the classification task well-defined, interpretable, and practically meaningful.

The project combines **exploratory data analysis**, **statistical reasoning**, **feature engineering**, **predictive modeling**, and **explainable AI** to answer three fundamental questions about startup survival.

---

## Repository Structure

```
Startup-Closing-Analysis-/
│
├── data/
│   ├── investments_VC.csv                  # Raw Crunchbase VC investment dataset (54,294 records)
│   ├── investments_VC_new.csv              # Cleaned and filtered dataset
│   └── model_df.csv                        # Final feature-engineered dataset for modeling
│
├── Scripts/
│   ├── data_cleaning.ipynb                 # Step 1 — Data loading, deduplication, type casting & quality validation
│   ├── data_eda.ipynb                      # Step 2 — Exploratory data analysis & business insights
│   ├── feature_Engineering.ipynb          # Step 3 — Market sector consolidation & derived features
│   └── preprocessing_and_modeling.ipynb   # Step 4 — Encoding, scaling, model training, evaluation & explainability
├── methodology.md                                       # In-depth statistical & ML methodology documentation
├── Startup_Outcome_Analysis_Methodology_Summary.docx   # Methodology summary report (Word format)
├── requirements.txt                                     # Python dependencies
└── README.md                                            # Project documentation (this file)
```

---

## System Requirements

| Requirement | Recommended Version |
|---|---|
| Python | 3.11.x |
| Jupyter Notebook / JupyterLab | Latest |
| pandas | >= 2.0 |
| scikit-learn | >= 1.3 |
| xgboost | >= 2.0 |
| matplotlib / seaborn | Latest |
| numpy | >= 1.24 |

> **Note:** A virtual environment (`venv/`) is included in the repository. It is recommended to activate it before running any notebooks.

---

## How to Run the Code

Follow these steps in order to reproduce all results.

### 1. Clone the Repository

```bash
git clone https://github.com/GaUrAnGjJ/Startup-Closing-Analysis-.git
cd Startup-Closing-Analysis-
```

### 2. Activate the Virtual Environment

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks in Order

Execute each notebook sequentially to reproduce the full pipeline:

| Step | Notebook | Purpose |
|------|----------|---------| 
| 1 | `Scripts/data_cleaning.ipynb` | Clean raw data, remove duplicates, fix data types |
| 2 | `Scripts/data_eda.ipynb` | Explore distributions and identify patterns |
| 3 | `Scripts/feature_Engineering.ipynb` | Engineer market sectors and create model features |
| 4 | `Scripts/preprocessing_and_modeling.ipynb` | Encode, scale, split data, train models, evaluate performance & interpret results |

---

## The Data Journey

### Where It All Begins

The raw dataset contained **54,294 startup records** with 39 features — spanning funding amounts, funding types, geographic data, market categories, founding dates, and more. Before any analysis could begin, the data needed to be thoroughly examined, corrected, and shaped into something trustworthy.

### Cleaning and Preparation

**Standardization** came first — column names had hidden leading and trailing whitespace that would cause silent bugs downstream. A single line of code resolved this, but it underscored an important principle: never assume your data is what it appears to be.

**Identifier columns** (`permalink`, `name`, `homepage_url`) were dropped early. These columns are unique per record, carry no predictive signal, and would only inflate model complexity without adding value.

**Temporal anomalies** revealed something interesting: 447 companies had a recorded first funding date before their official founding date. Rather than blindly deleting these records, we investigated. Most reflected a real-world phenomenon — startups often receive angel or seed funding before formal legal incorporation. Only records with a single funding round where the funding gap exceeded five years were treated as genuine errors and removed.

**Target variable scoping** was critical. The dataset contained four status categories:

- `acquired` — reached a successful exit
- `closed` — shut down operations
- `operating` — still active (future unknown)
- `NaN` — unknown status

For a supervised learning task, we need known, final outcomes. We retained only **acquired** (3,692 companies) and **closed** (2,603 companies) records — a total of **6,295 labeled examples** — and preserved the rest for potential future inference.

---

## What the Data Revealed

### The Funding Story

Exploratory analysis uncovered a clear and consistent pattern: **acquired startups raise substantially more money than those that close**. This relationship held across total funding amounts, number of funding rounds, and funding duration.

The funding distribution is highly right-skewed — most startups raise modest amounts through one or two rounds, while a small number attract enormous investments. Approximately **60% of startups raised funding only once**, and **81% raised funding at most twice**. Only **0.1%** ever reached 10 or more funding rounds.

Yet it is precisely those multi-round companies that disproportionately end up acquired. Sustained fundraising signals investor confidence, operational longevity, and market traction — all of which are strongly tied to eventual acquisition.

### The Market Sector Problem

The raw dataset contained over **350 distinct market categories** — far too many for a model to learn meaningful patterns from. A startup labeled "Social Commerce" and another labeled "E-commerce" are fundamentally similar, but the model would treat them as entirely unrelated.

To address this, we applied domain knowledge to consolidate all 350+ categories into **17 coherent business sectors** (plus an "Other" bucket accounting for 8.44% of miscellaneous entries). Groups such as Software, SaaS, Cloud Computing, and Enterprise Software were merged into a single **Software & Cloud** sector. This reduced noise, improved interpretability, and gave the model a more meaningful signal to learn from.

---

## Dashboard Visualization

An interactive Tableau dashboard accompanies this analysis, providing visual exploration of the funding landscape, sector distributions, and outcome patterns across acquired and closed startups.

**[📊 View Interactive Dashboard on Tableau Public](https://public.tableau.com/views/dashboard_17832416309830/Dashboard2?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

The dashboard enables stakeholders to:
- Explore how funding amount and round count vary across acquisition outcomes
- Compare market sector distributions between acquired and closed startups
- Identify geographic and temporal patterns in startup exits
- Filter interactively by sector, country, and funding type

> The Tableau workbook (`dashboard.twb`) is also included in the repository for local exploration.

---

## Model Development and Selection

### Candidates Evaluated

Three classification models were trained and evaluated using **stratified 5-fold cross-validation**:

1. **Logistic Regression** — interpretable, stable, coefficient-based
2. **Random Forest** — ensemble of decision trees
3. **XGBoost** — gradient boosting, strong performance on tabular data

### Experiment Results Comparison Summary

The following table presents the full evaluation results for all three models trained using stratified 5-fold cross-validation, ranked by ROC-AUC:

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 1 | **Logistic Regression** | **0.6966** | **0.6865** | 0.4885 | **0.5708** | **0.7366** |
| 2 | XGBoost | 0.6910 | 0.6329 | **0.6000** | **0.6160** | 0.7325 |
| 3 | Random Forest | 0.6759 | 0.6167 | 0.5692 | 0.5920 | 0.7071 |

**Key finding:** Logistic Regression achieved the **highest ROC-AUC (0.7366)**, **highest accuracy (69.66%)**, and **highest precision (0.6865)**, making it the best overall discriminator between acquired and closed startups. XGBoost achieved a higher recall and F1-score by predicting more acquisitions, but at the cost of more false positives and lower overall discrimination. Random Forest ranked last across all five metrics and was eliminated from further consideration.

### Why Logistic Regression Was Selected

The project's core objective was not only to predict outcomes but to **understand** them. Logistic Regression provides direct, coefficient-based explanations — each feature's contribution is transparent and interpretable. For an analytics project centered on understanding *why* certain startups get acquired, this interpretability made Logistic Regression the appropriate final model.

Hyperparameter tuning delivered a modest but consistent improvement across all metrics, and the tuned Logistic Regression was selected as the **final model**.

---

## Research Questions and Findings

---

### Q. Which startup characteristics are most strongly associated with successful acquisition?

**A.** The logistic regression model identified **Market Sector** as the most influential group of characteristics, indicating that the industry in which a startup operates is strongly associated with whether it is ultimately acquired or closed.

Beyond sector, the following characteristics were also positively associated with acquisition:

- **Higher total funding** — more capital raised correlates with better long-term outcomes
- **Longer fundraising duration** — sustained investor engagement signals continued confidence
- **Venture funding** — receiving venture capital is a strong positive indicator of acquisition

These findings suggest that acquisition is driven by a combination of *what a startup does* (market sector) and *how well-funded it is* (capital access and investor backing).

---

### Q. Which funding type has the strongest association with acquisition?

**A.** Among all funding mechanisms, **venture capital** has the strongest association with acquisition.

It held the **largest relative contribution among all funding-type variables (2.72%)** and carried a **negative logistic regression coefficient for closure** — meaning that venture-backed startups had lower odds of shutting down and, conversely, higher odds of being acquired.

This relationship held true after controlling for total funding amount, market sector, fundraising duration, and all other funding sources. Venture capital is not simply a proxy for more money — it carries signal about investor validation, board oversight, strategic networks, and the pressure to reach a liquidity event, all of which make acquisition a more likely outcome.

---

### Q. Does market sector matter?

**A.** Yes — decisively.

The analysis indicates that a startup's market sector is one of the **single strongest characteristics** associated with whether it is acquired or closed. The logistic regression model consistently assigned substantial weight to sector variables, suggesting that industry choice plays a fundamental role in startup outcomes.

**Market sectors most strongly associated with acquisition:**

| Rank | Sector |
|------|--------|
| 1 | AI & Data |
| 2 | Security |
| 3 | Financial Services |
| 4 | Travel & Hospitality |
| 5 | Education |

These sectors share common traits: they attract strategic acquirers with significant capital, they contain intellectual property worth acquiring, and they operate in markets where consolidation is a well-established growth strategy.

---

## Conclusion

What began as a question — *can we predict which startups get acquired?* — evolved into a thorough analytical journey through data cleaning, exploratory analysis, feature engineering, and machine learning.

The answer is nuanced but clear: **certain characteristics do systematically differentiate startups that get acquired from those that close**. Market sector, venture capital backing, total funding raised, and fundraising duration are the most powerful signals. A startup operating in AI & Data or Security, backed by venture capital, with sustained multi-round fundraising, is meaningfully more likely to be acquired than one lacking these characteristics.

The final tuned Logistic Regression model achieved a **ROC-AUC of 0.7396** — a meaningful result given the inherent unpredictability of startup outcomes, where timing, macroeconomic conditions, and competitive dynamics always play a role that no model can fully capture.

This project demonstrates that even with a focused dataset and an interpretable model, data science can extract actionable insight from what might otherwise appear to be a highly unpredictable domain.

---

## Data Source

- **Dataset:** Crunchbase VC Investments Dataset
- **Source file:** `data/investments_VC.csv`
- **Records:** 54,294 startups
- **Encoding:** Latin-1

---

*Built with Python, scikit-learn, XGBoost, and Jupyter.*
