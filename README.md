# Customer Churn Prediction

A notebook-based B.Tech Predictive Analytics project using the IBM Telco Customer Churn sample. The workflow cleans the data, compares Logistic Regression, Decision Tree, and Random Forest with validation-based selection, and serves the selected model through a Panel dashboard.

The model estimates the historical churn class and churn probability from customer attributes. The dataset is cross-sectional and does not predict when a customer will leave.

## Project structure

```text
Customer_Churn_Prediction/
├── Datasets/
│   ├── Telco-Customer-Churn.csv
│   └── Telco-Customer-Churn_Cleaned.csv
├── 01_Data_Cleaning.ipynb
├── 02_Model_Training.ipynb
├── 03_Visualization_and_Dashboard.ipynb
├── app.py
├── selected_model.joblib
├── Research_Paper.docx
├── requirements.txt
└── README.md
```

## Run the notebooks

Use Python 3.14.4 with the packages in `requirements.txt`. Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

From this project folder, run the notebooks in order:

1. `01_Data_Cleaning.ipynb` checks the raw data and writes `Datasets/Telco-Customer-Churn_Cleaned.csv`. The original CSV remains unchanged.
2. `02_Model_Training.ipynb` performs EDA and validation-based model selection, evaluates all three models on a held-out test set, and writes one `selected_model.joblib` bundle. The bundle includes the fitted pipeline and compact test summaries used by the dashboard.
3. `03_Visualization_and_Dashboard.ipynb` previews the existing Panel dashboard and checks its risk filter and individual prediction callback. The full dashboard implementation stays in `app.py`.

The preprocessing and split settings are kept consistent across the notebook, saved pipeline, and dashboard: stratified 80/20 train-test split, random state 42, a 25% validation slice from the training partition, balanced class weights, training-only imputation and encoding, and a default classification threshold of 0.50. Model selection uses validation F1-score, with recall and ROC-AUC as tie-breakers. The held-out test set is reserved for final comparison.

## Run the dashboard

Run notebooks 01 and 02 first so the cleaned dataset and selected model bundle exist. Then, from the project folder:

```powershell
panel serve app.py --show
```

The dashboard retains the business overview, interactive churn charts, model comparison, confusion matrix, feature importance, filterable held-out customer risk table, and individual customer prediction form.

## Current reference results

The existing project run selected Random Forest on the validation split. Its held-out test scores were accuracy 0.773, precision 0.553, recall 0.751, F1-score 0.637, and ROC-AUC 0.842. Decision Tree achieved the highest test recall at 0.794. Running notebook 02 regenerates and displays the actual results for the current environment.

## Data and interpretation notes

The source data contain 7,043 rows and 21 columns. Eleven blank `TotalCharges` strings occur for zero-tenure customers; they are represented as zero. No rows are removed. `customerID` is retained for reporting and excluded from model features. Global feature importance describes model reliance, not correlation or causation. The dataset is fictional and results may not generalize to a live telecom population. Probability thresholds for real retention campaigns should be selected using campaign costs and capacity.

The paper is an academic draft for independent review and revision before submission.
