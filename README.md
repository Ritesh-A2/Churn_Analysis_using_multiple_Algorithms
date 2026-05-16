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
To ensure clean data pipeline delivery, the following steps were executed using automated automation pipelines:
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
* **XGBoost Classifier:** Advanced high-performance gradient boosting.

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
Models were compared using **Accuracy, Precision, Recall, and F1-Score**.

| Model Type | Key Insights & Performance |
| :--- | :--- |
| **Traditional Models** | Provide baseline performance; Logistic Regression offered high interpretability. |
| **Boosting (XGBoost/GBM)** | Significantly reduced false negatives (improved Recall for churners). |
| **Stacking Classifier** | **Achieved the best overall performance** by combining strengths of individual models. |

---

## ⚙️ Technologies Used
* **Languages & Core:** Python, NumPy, Pandas
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Deployment/UI:** Streamlit

---

## 🌐 Model Deployment
The final pipeline is structured for real-time predictions and can be deployed using:
* **Streamlit** (Interactive Frontend)


---

## 🚀 Future Scope
* Implementing Deep Learning Architectures (MLPs).
* Designing real-time customer monitoring and automated retention trigger systems.
* Cloud deployment with Explainable AI (XAI) integration for model transparency.
