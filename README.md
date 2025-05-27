# Bank Marketing Campaign Analysis: Predicting Term Deposit Subscription

This analysis uses the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) to predict whether a customer will subscribe to a term deposit. The project follows the **CRISP-DM framework** and explores various machine learning approaches for this binary classification task.

See the detailed step-by-step model analysis [check the analysis notebook](prompt_III.ipynb)

## CRISP-DM Steps

### 1. Business Understanding

- **Goal:** Identify patterns and features that indicate which customers are likely to subscribe to a term deposit.
- **Value:** Improve marketing campaign targeting, increase subscriptions, and reduce customer annoyance from repeated contact.

---

### 2. Data Understanding

- **Dataset:** Bank marketing campaign data from a Portuguese banking institution.
- **Features:** Includes demographics (age, job, marital, education), campaign contact history, economic context, and previous campaign outcomes.
- **Target:** `y` – whether the customer subscribed ("yes"/"no") to a term deposit.

---

### 3. Data Preparation

- **Missing/Unknown Values:** Handled "unknown" values by encoding them as a new category (`other`) and creating indicator features.
- **Feature Engineering:**  
    - Created age bins, new customer indicators, and campaign success rate features.
    - Encoded cyclical features for months.
    - One-hot encoded categorical variables.
- **Scaling:** All features were scaled using `StandardScaler` for distance-based models.
- **Class Imbalance:** Noted that only ~11% of customers subscribed; most models are biased towards the majority class.

---

### 4. Modeling

- **Baseline:** Used the majority class as a naive baseline (~89% accuracy).
- **Logistic Regression:**  
    - Provided interpretability through coefficients.
    - Achieved high overall accuracy (90%), but low recall (19%) for "yes" class.
- **K-Nearest Neighbors (KNN):**  
    - Hyperparameters tuned with `GridSearchCV`.
    - Achieved a modest F1-score for "yes" class (0.38), but recall remains low.
- **Decision Tree:**  
    - Showed which features are most important for prediction.
    - High accuracy (84%) and interpretability, but still struggles with the minority class.
- **Support Vector Machine (SVM):**
    - Applied extensive feature engineering, scaling, and hyperparameter tuning (`C`, `kernel`, `gamma`).
    - Used class balancing to improve recall for the minority class.
    - Best-tuned SVM (with Manhattan distance and 3 neighbors) achieved ~89% test accuracy and F1-score of 0.38 for "yes" class, with precision of 0.50 and recall of 0.30.
    - SVM model performance was similar to KNN: strong at detecting "no", but recall for "yes" (minority) remained challenging.
- **Feature Importance Analysis:**  
    - Key positive predictors: positive economic context (`euribor3m`, `emp.var.rate`), prior campaign success (`poutcome=success`), older age, and higher education.
    - Key negative predictors: loan/default status, failed previous campaigns, and negative economic sentiment.

| Model	|  Accuracy	 | Precision (yes)|  Fit Time (s) |
| ----- | ---------- | -------------  | ------------  |
|Logistic Regression |	0.896455 | 0.638514	|	1.170570 |
|KNN	| 0.885530	| 0.491342	| 0.003748|
|Decision Tree|	0.836489 |	0.305660 |		0.183043
|SVM	| 0.886744	|1.000000	| 0.002139	|
---

### 5. Evaluation

- **Metrics:** Focused on accuracy, precision, recall, and F1-score for the minority class.
- **Imbalanced Data Challenge:** All models predict "no" well, but struggle to capture most "yes" cases (low recall/F1).
- **Feature Engineering Impact:** Enhanced features slightly improved minority class prediction, but class imbalance remains a challenge.

---

## Key Takeaways

- **Who is likely to subscribe?**
    - Customers with positive economic indicators, successful prior contact, and certain demographic profiles (e.g., older, more educated) are more likely to subscribe.
    - Negative previous outcomes, existing loans/defaults, and high campaign contacts reduce likelihood.
- **Model performance:**  
    - All models have high overall accuracy driven by the majority class ("no"), but have difficulty predicting the minority class ("yes"). SVM, like KNN, achieved modest gains for "yes" using hyperparameter tuning and class balancing, but recall remains a challenge.
- **Next Steps:**
    - Address class imbalance using resampling or class weights.
    - Explore ensemble methods for improved minority class detection.
    - Further optimize features and perform hyperparameter tuning.


---

## References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [CRISP-DM Data Science Framework](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
