# ShopSense AI

## Online Shopper Purchase Intention Prediction Using Supervised Machine Learning

**Tagline:** Understand Customer Behaviour. Predict Purchase Intent.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-17%20passed-22a06b)
![License](https://img.shields.io/badge/license-MIT-blue)

ShopSense AI is an academic machine-learning mini project that predicts whether an online browsing session is likely to result in a purchase. It combines three supervised classifiers with a Streamlit dashboard, real held-out evaluation, interactive visualizations, and deterministic examples from the test set.

## Contents

- [Quick Start](#quick-start)
- [Introduction and Problem Statement](#introduction-and-problem-statement)
- [Objectives and Real-World Use Case](#objectives-and-real-world-use-case)
- [Dataset](#dataset-description)
- [Machine Learning Workflow](#machine-learning-algorithms)
- [Actual Results](#actual-results)
- [Frontend Features](#frontend-features)
- [Installation and Execution](#installation-and-execution-windows)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Limitations](#limitations)

## Quick Start

```powershell
git clone https://github.com/Aashin-A-P/Shop_Sense_AI.git
cd Shop_Sense_AI
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The trained artifacts are included, but running `train_model.py` reproduces them from the local CSV and refreshes `models/metadata.json`.

## Introduction and Problem Statement

E-commerce sites receive many visits, while comparatively few sessions end in purchases. Manually analyzing thousands of browsing sessions is slow and difficult. ShopSense AI learns patterns from historical session data and estimates purchase intention from page visits, time spent, engagement rates, visitor information, and technical session attributes. It is an academic analytics prototype, not a live commerce or customer-tracking system.

## Objectives and Real-World Use Case

- Analyze historical shopping behaviour and purchase patterns.
- Train Logistic Regression, K-Nearest Neighbors, and Decision Tree classifiers.
- Compare accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Predict purchase intention and positive-class probability for a new session.
- Visualize dataset patterns, model performance, and feature importance.

An e-commerce administrator enters session characteristics. The selected pipeline returns `Purchase` or `No Purchase` and an estimated probability. No shopping cart, payment, authentication, database, or real-time integration is included.

## Dataset Description

- **Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Local file:** `online_shoppers_intention.csv`
- **Records:** 12,330
- **Input features:** 17
- **Target:** `Revenue` (`True` = purchase, `False` = no purchase)
- **Class distribution:** 1,908 purchase and 10,422 non-purchase sessions
- **Missing values:** 0
- **Duplicate rows:** 125 retained

The source CSV is never modified. Duplicate rows are retained because identical feature profiles can represent distinct shopping sessions, and no session identifier proves that they are accidental copies.

## Dataset Features

| Feature | Meaning / type |
|---|---|
| Administrative | Administrative pages visited |
| Administrative_Duration | Seconds spent on administrative pages |
| Informational | Informational pages visited |
| Informational_Duration | Seconds spent on informational pages |
| ProductRelated | Product-related pages visited |
| ProductRelated_Duration | Seconds spent on product-related pages |
| BounceRates | Proportion of visits leaving from a page |
| ExitRates | Proportion of page views ending a session |
| PageValues | Average page value before a transaction |
| SpecialDay | Proximity to a special shopping day |
| Month | Month category |
| OperatingSystems | Operating-system category code |
| Browser | Browser category code |
| Region | Geographic region category code |
| TrafficType | Traffic-source category code |
| VisitorType | New, returning, or other visitor |
| Weekend | Whether the session occurred on a weekend |

## Machine Learning Algorithms

1. **Logistic Regression** uses the logistic function to estimate a binary outcome probability. It uses `max_iter=1000`, balanced class weights, and random state 42.
2. **K-Nearest Neighbors (KNN)** classifies a session using nearby scaled training examples. It uses five neighbors.
3. **Decision Tree** learns rules through recursive splits. It uses maximum depth 8, minimum leaf size 10, balanced class weights, and random state 42.

This connects directly to the syllabus: **Unit I** covers Logistic Regression, while **Unit II** covers K-Nearest Neighbors and Decision Trees.

## Data Preprocessing

- `Revenue` is converted to binary integers.
- Numeric features are median-imputed for pipeline robustness.
- `Month`, `VisitorType`, and four integer category codes are one-hot encoded with unknown-category handling.
- `Weekend` is passed as a Boolean binary feature.
- Numeric inputs are standardized for Logistic Regression and KNN; Decision Tree inputs are not scaled.
- Each saved scikit-learn `Pipeline` contains its fitted `ColumnTransformer` and classifier.

Encoding converts categories into machine-readable indicators. Scaling puts numeric measurements on comparable ranges, which is important for distance-based KNN. Preprocessing is fitted only on training data, preventing leakage from the held-out set.

## Training Methodology

The project uses one shared, stratified 80/20 split with `random_state=42`:

- Training: 9,864 records (8,338 no-purchase / 1,526 purchase)
- Testing: 2,466 records (2,084 no-purchase / 382 purchase)

A train-test split measures behaviour on records not used for fitting. Stratification preserves the imbalanced class ratio in both partitions.

## Evaluation Metrics

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted purchases that are actual purchases.
- **Recall:** proportion of actual purchases that are detected.
- **F1-score:** harmonic mean of positive-class precision and recall.
- **ROC-AUC:** ability of probabilities to rank purchase sessions above non-purchase sessions across thresholds.
- **Confusion matrix:** true-negative, false-positive, false-negative, and true-positive counts.

Because purchases are the minority class, positive-class F1-score is the primary metric; accuracy alone can conceal weak purchase detection.

## Actual Results

These values were produced by `python train_model.py` from the included CSV and saved to `models/metadata.json`.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8414 | 0.4922 | 0.7435 | 0.5923 | 0.8931 |
| K-Nearest Neighbors | 0.8715 | 0.6419 | 0.3848 | 0.4812 | 0.7572 |
| **Decision Tree** | **0.8309** | **0.4738** | **0.8298** | **0.6032** | **0.9033** |

Decision Tree is the default because it has the highest held-out positive-class F1-score (0.6032). This does not mean it is universally best. Selecting with test-set results is an exploratory academic comparison, not an unbiased final estimate. Production work should use cross-validation plus a separate untouched final test set.

## System Architecture

```text
Local CSV
   ├── validation and shared stratified split
   ├── preprocessing → Logistic Regression pipeline → .pkl
   ├── preprocessing → KNN pipeline                 → .pkl
   └── preprocessing → Decision Tree pipeline       → .pkl
                                      │
                              metadata.json
                                      │
                 Streamlit Dashboard / Prediction / Comparison
```

## Frontend Features

- **Dashboard:** live metrics, purchase donut, visitor and monthly conversion charts, browsing-duration box plot, and dataset facts.
- **Purchase Prediction:** all 17 inspectable inputs, dataset-derived defaults, model selector, model probability gauge, validation, and reset.
- **Model Comparison:** actual metric table, F1 chart, selectable confusion matrices, top ten fitted Decision Tree importances, and algorithm explanations.
- Cached dataset/model loading and clear errors for missing or invalid artifacts.

## Sample Customer Feature

The prediction page can load one purchase and one no-purchase record from the deterministic held-out partition. Selection uses only the recorded class—not whether a model predicts it correctly. All 17 original values populate the editable form and can be evaluated with any model. The app reports the predicted outcome, recorded outcome, and whether they match. Editing any input clears the recorded-outcome comparison to avoid a misleading label.

## Installation and Execution (Windows)

Python 3.11 or newer is recommended.

```powershell
cd D:\Shop_Sense_AI
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
python -m pytest -v
python -m streamlit run app.py
```

Open <http://localhost:8501> if the browser does not open automatically. The app deliberately does not retrain at startup; if artifacts are absent, run the training command first.

## Testing

Run the complete verification suite from the repository root:

```powershell
python -m pytest -v
```

The suite checks the dataset schema, stratified split, non-overlapping indices, all saved pipelines, class probabilities, held-out sample loading, metadata, confusion matrices, and Decision Tree feature importance. The latest verified run completed with **17 passed tests**.

## Project Structure

```text
Shop_Sense_AI/
├── online_shoppers_intention.csv
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   └── metadata.json
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── predictions.py
└── tests/
    └── test_pipeline.py
```

## Connection to the Machine Learning Syllabus

ShopSense AI demonstrates **supervised learning** because each historical row has a known `Revenue` label. The problem is binary **classification**. It covers categorical feature encoding, numeric scaling, stratified train-test splitting, probability prediction, class imbalance, and evaluation. It gives runnable examples of Unit I Logistic Regression and Unit II KNN and Decision Trees.

## Limitations

- Results describe one historical UCI dataset and may not generalize to another store or time period.
- The target is imbalanced, and business-specific classification costs are not available.
- Category codes such as browser and region have no semantic labels in the CSV.
- Feature importance is predictive association, not causation.
- One held-out split is used for a simple course demonstration.
- Test-set comparison is used for exploratory selection; there is no second untouched final set.
- The offline app does not track live customers.

## Future Enhancements

- Add cross-validation and a final untouched evaluation set.
- Compare probability calibration, threshold tuning, and imbalance strategies.
- Add permutation importance or SHAP if the academic scope expands.
- Evaluate other interpretable ensemble methods.
- Add anonymized batch scoring and model-drift reporting.

## Reproducibility

The split and applicable estimators use random state 42. Metadata records configurations, library version, class counts, confusion matrices, and metrics. Compatible library versions and the same dataset should reproduce the reported results.

## License

This project is distributed under the [MIT License](LICENSE).
