# **Heart Disease Prediction Using Logistic Regression (Framingham Heart Study Dataset)**



### **Overview of the Project**



This project predicts the 10-year risk of Coronary Heart Disease (CHD) using the Framingham Heart Study dataset. The model uses Logistic Regression to analyze important medical and lifestyle features that influence heart disease.



A simple Graphical User Interface (GUI) built using Tkinter allows users to input patient details and instantly view:

* Model prediction (0 = No Heart Disease Risk, 1 = High Risk)
* Probability score
* ROC Curve
* Trained model accuracy metrics



This project demonstrates data preprocessing, model training, evaluation, and deployment in GUI form.



### **Features**

1. Machine Learning

* Logistic Regression Model
* Automatic preprocessing (missing value handling, scaling)
* Train-test split
* Cross-validation
* ROC-AUC evaluation
* Confusion matrix \& classification report



2\. GUI (Tkinter)

* Input fields for all patient features
* Pre-filled values using dataset medians
* Prediction output with probability
* Show ROC Curve
* Save trained model



3\. Dataset

* Uses Framingham Heart Study Dataset (framingham.csv)
* Predicts TenYearCHD



### **Technologies / Tools Used**

1. Machine Learning

* Python 3
* pandas, numpy
* scikit-learn
* matplotlib



2\. GUI

* Tkinter
* ttk Widgets



3\. Model Saving

* joblib



### **Steps to Install \& Run the Project**

1. Install Python (if not installed)

2\. Install required libraries

Run in Command Prompt / Terminal:

pip install pandas numpy scikit-learn matplotlib joblib

3\. Place your dataset

Copy framingham.csv to:

C:\\Users\\Atieksh\\Downloads\\

OR update the path inside the script:

DATA\_PATH = r"C:\\path\\to\\framingham.csv"

4\. Run the Project

Save the Python script as:

heart\_gui.py

Then run:

python heart\_gui.py



### **Instructions for Testing the Application**

1. Open the GUI

The window will display:

* Dataset statistics
* Model metrics
* Input fields for 15 feature



2\. Enter Patient Test Values

You can manually type values or keep default medians.

Example test input:

male: 1

age: 54

education: 2

currentSmoker: 1

cigsPerDay: 15

BPMeds: 0

prevalentStroke: 0

prevalentHyp: 1

diabetes: 0

totChol: 233

sysBP: 140

diaBP: 90

BMI: 27.5

heartRate: 78

glucose: 85



3\. Click “Predict from inputs”

You will see:

Prediction: 0 or 1

Probability: 0.xxx



4\. Click “Show ROC Curve”

The ROC curve will open in a new window.



5\. The model will be saved as:

framingham\_gui\_model.joblib



