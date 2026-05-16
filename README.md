# 🧠 Customer Churn Prediction Using Machine Learning


## ❗ Problem Statement
Customer churn is one of the biggest challenges faced by subscription-based and service-oriented businesses. Churn occurs when customers stop using a company's products or services, leading to revenue loss and reduced customer retention.

The major challenge is to identify customers who are likely to leave the company before churn actually happens. Early prediction helps businesses take preventive actions such as personalized offers, customer engagement, and support improvements.

### 🎯 Project Objectives
* **Analyze** customer behavior and service usage patterns.
* **Identify** factors responsible for customer churn.
* **Predict** whether a customer will churn or not.
* **Compare** multiple machine learning algorithms.
* **Improve** retention strategies using data-driven insights.
* **Discover** hidden customer groups using clustering algorithms.

---

## 🌐 Model Deployment & Live Demo
The trained model is fully deployed and available for real-time predictions. You can interact with the live application here:

👉 **[Live Streamlit Web App](https://churnanalysisusingmultiplealgorithms-je7wvbedzjzg3pgkmsnlbd.streamlit.app/)**

### 🛠️ How to run locally:
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---

## 📌 Dataset Overview
This dataset contains customer information collected from a telecom/service-based company. It includes detailed information about customers' demographics, subscription details, service usage, billing information, and account history.

### 🗂️ Key Features
* **Demographics:** Gender, Senior Citizen, Partner, Dependents
* **Services:** Tenure, Phone Service, Internet Service, Online Security, Tech Support
* **Billing & Account:** Contract Type, Payment Method, Monthly Charges, Total Charges
* **Target Variable:** `Churn` (Yes/No - whether the customer left or stayed)

### 📊 Capabilities
* Exploratory Data Analysis (EDA) & Data Visualization
* Supervised Machine Learning & Ensemble Learning
* Unsupervised Clustering Analysis

---

## 📊 Exploratory Data Analysis (EDA)
EDA was performed to understand the structure and behavior of the dataset before model building.

* **Structure & Quality Check:** Analyzed missing values, duplicates, and data types.
* **Feature Analysis:** Visualized distributions of numerical features and proportions of categorical classes.
* **Correlation & Outliers:** Investigated multi-collinearity among billing metrics and handled extreme values.
* **Insights Gained:** Identified high-risk customer groups and structural patterns affecting long-term retention.

---

## 🐼 Data Preprocessing
To ensure clean data pipeline delivery, the following steps were executed using automated pipelines:
1. **Handling Missing Values** & cleaning inconsistent entries.
2. **Removing Duplicates** to prevent data leakage.
3. **Feature Encoding** for categorical variables via Column Transformers.
4. **Feature Scaling** to normalize numerical variations.

---

## 🤖 Machine Learning Models Implemented

### 🔹 Supervised Learning (Base Models)
* **Logistic Regression:** Predicts churn probability using logistic functions.
* **Decision Tree Classifier:** Splits data into explicit decision rules.
* **Random Forest Classifier:** Combines multiple independent trees to reduce overfitting.
* **K-Nearest Neighbors (KNN):** Distance-based neighborhood classification.
* **Support Vector Machine (SVM):** Optimizes hyperplanes for crisp class separation.
* **Naive Bayes:** Probabilistic classification based on Bayes’ theorem.

### 🚀 Ensemble Learning (Advanced)
* **Bagging:** Uses multiple estimators trained on random subsets.
* **AdaBoost & Gradient Boosting:** Sequential learning minimizing residual errors.
* **Voting Classifier:** Aggregates predictions across diverse base estimators.
* **Stacking Classifier:** Passes base model predictions to a meta-model for final output.

### 🌀 Unsupervised Learning
* **K-Means Clustering:** Groups customers by usage similarities.
* **DBSCAN:** Density-based grouping to discover spatial clusters and detect anomalies.

---

## 📈 Model Comparison & Evaluation
All models were thoroughly evaluated using **Accuracy, Precision, Recall, and F1-Score**. Below is the comprehensive performance comparison dashboard:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8211** | 0.6861 | 0.5978 | **0.6389** |
| **Stacking** | 0.8161 | 0.6862 | 0.5630 | 0.6185 |
| **SVM** | 0.8140 | **0.7003** | 0.5201 | 0.5969 |
| **AdaBoost** | 0.8133 | 0.6821 | 0.5522 | 0.6103 |
| **Voting** | 0.8097 | 0.6966 | 0.4986 | 0.5812 |
| **Gradient Boosting** | 0.8090 | 0.6721 | 0.5442 | 0.6014 |
| **Random Forest (Tuned)** | 0.8048 | 0.6828 | 0.4906 | 0.5709 |
| **Random Forest (Base)** | 0.7920 | 0.6492 | 0.4664 | 0.5429 |
| **Bagging** | 0.7799 | 0.6225 | 0.4289 | 0.5079 |
| **Decision Tree** | 0.7154 | 0.4623 | 0.4611 | 0.4617 |
| **Naive Bayes** | 0.6657 | 0.4358 | **0.8927** | 0.5857 |

### 💡 Key Performance Insights:
* **Best All-Rounder:** **Logistic Regression** achieved the highest overall **Accuracy (82.11%)** and **F1-Score (63.89%)**, making it the most balanced model for this dataset.
* **Ensemble Power:** Advanced techniques like **Stacking** and **AdaBoost** performed exceptionally close to the top, showing strong stability with accuracy over **81.3%**.
* **Highest Churn Detection (Recall):** **Naive Bayes** delivered a massive **Recall of 89.27%**. If a business wants to ensure absolutely no churner goes unnoticed (minimizing false negatives), this model catches the most risk, though with a trade-off in precision.
* **Most Precise:** **SVM** led with a **Precision of 70.03%**, meaning its flags for customer churn are the most reliable.

---

## ⚙️ Technologies Used
* **Languages & Core:** Python, NumPy, Pandas
* **Machine Learning:** Scikit-learn, Ensemble Methods
* **Visualization:** Matplotlib, Seaborn
* **Deployment/UI:** Streamlit

---

## 🚀 Future Scope
* Implementing Deep Learning Architectures (MLPs) to study sequence behavior.
* Designing real-time customer monitoring and automated retention trigger systems.
* Cloud deployment with Explainable AI (XAI) integration for model transparency.
