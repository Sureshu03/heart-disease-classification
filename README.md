# heart-disease-classification
The Heart Disease Classification App is an interactive web application built so that enables users to evaluate the likelihood of heart disease using multiple machine learning models.

**Problem Statement**
Heart disease is one of the leading causes of death worldwide. Early detection and accurate classification of patients at risk can significantly improve treatment outcomes.
This project builds a Streamlit-based web application that allows users to classify heart disease using multiple machine learning models. The app provides predictions and evaluation metrics to compare model performance.

**Dataset Description**
The dataset contains patient health attributes such as age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, ST depression, slope of ST segment, number of major vessels, and thalassemia type.

**Output**
If the Target value is 1, it indicates the presence of Heart disease. 
If the Target value is 0, it indicates the absence of Heart disease.

**Models Used**
| ML Model Name       | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|---------------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.86     | 0.92 | 0.83      | 0.90   | 0.87 | 0.72 |
| Decision Tree       | 1.00     | 1.00 | 1.00      | 0.99   | 1.00 | 0.99 |
| kNN                 | 0.88     | 0.95 | 0.90      | 0.88   | 0.89 | 0.77 |
| Naïve Bayes         | 0.83     | 0.91 | 0.81      | 0.87   | 0.84 | 0.66 |
| Random Forest       | 1.00     | 1.00 | 1.00      | 0.99   | 1.00 | 0.99 |
| XGBoost             | 1.00     | 1.00 | 1.00      | 0.99   | 1.00 | 0.99 |


**Observation**
| ML Model Name       | Observation about model performance                                                                           |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Good balance of precision and recall, interpretable coefficients, slightly lower MCC compared to ensembles.   |
| Decision Tree       | Achieved perfect accuracy and AUC, but may be overfitting; performance could drop on unseen data.             |
| kNN                 | Solid performance with high precision and recall, but sensitive to scaling and choice of k.                   |
| Naïve Bayes         | Fast and simple, decent metrics, but lower MCC and precision compared to other models.                        |
| Random Forest       | Excellent ensemble performance, robust and less prone to overfitting, very high accuracy and MCC.             |
| XGBoost             | Outstanding ensemble performance, matches Random Forest with perfect metrics, strong generalization expected. |
