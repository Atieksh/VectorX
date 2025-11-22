### **Problem Statement**



Heart diseases are a major global health concern and one of the leading causes of death worldwide. Early identification of individuals who are at high risk of developing heart disease enables timely treatment and preventive measures.



Medical professionals often rely on multiple health indicators—such as blood pressure,

cholesterol levels, smoking habits, and glucose levels—to assess this risk.



This project aims to develop a machine learning–based predictive system that can accurately estimate the 10-year risk of Coronary Heart Disease (CHD) using patient data from the Framingham Heart Study.

The objective is to support healthcare practitioners with a data-driven tool to improve

decision-making.



### **Scope of the Project**



This project includes:



1. Data Processing

* Handling missing values
* Normalizing and scaling numeric features
* Splitting dataset into training and testing sets



2\. Predictive Modeling

* Building a Logistic Regression model
* Model evaluation using Accuracy, ROC-AUC, Precision, Recall, and F1-Score
* Generating ROC curve for performance visualization



3\. Graphical User Interface (GUI)

* Tkinter-based interactive application
* User inputs for all health-related features
* Real-time prediction display
* Probability score for risk estimation
* Ability to save the trained machine learning model



4\. Visualization

* Show ROC curve for the trained model directly from GUI



### **Target Users**

1. Healthcare Practitioners

Doctors, nurses, and health workers who wish to quickly estimate a patient’s heart disease risk.



2\. Researchers

Medical research teams studying heart disease patterns and risk factors.



3\. Students

ML and data science learners studying classification algorithms and GUI applications.



4\. Developers

Programmers looking to implement ML models with GUI-based deployment.



### **High-Level Features**

1. Machine Learning Features

* Logistic Regression–based binary classification
* Balanced class handling
* Cross-validation for reliable metrics
* Model saving using Joblib



2\. GUI Features

* Tkinter interface for easy interaction
* Scrollable input form for 15 health indicators
* Auto-filled median values for convenience
* Prediction results displayed with probability score
* Button to visualize ROC curve
* Option to save trained model



3\. Dataset Features

* Uses the Framingham Heart Study dataset (framingham.csv)
* Predicts TenYearCHD risk
