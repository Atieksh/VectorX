import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import joblib

DATA_PATH = r"C:\Users\Atieksh\Downloads\framingham.csv"
MODEL_OUTPUT = "framingham_gui_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET = "TenYearCHD"

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise SystemExit(f"Failed to load dataset at {DATA_PATH}: {e}")

FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES].copy()
y = df[TARGET].copy()

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_cols)], remainder="drop")

clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="roc_auc")

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

medians = X.median().to_dict()

def save_model():
    joblib.dump(clf, MODEL_OUTPUT)
    messagebox.showinfo("Saved", f"Model saved to {MODEL_OUTPUT}")

def predict_from_inputs():
    try:
        data = {}
        for f in FEATURES:
            val = entries[f].get().strip()
            if val == "":
                data[f] = medians[f]
            else:
                data[f] = float(val)
        row = pd.DataFrame([data])
        pred = clf.predict(row)[0]
        prob = clf.predict_proba(row)[:, 1][0]
        lbl_result.config(text=f"Prediction: {int(pred)}  Probability: {prob:.4f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input or prediction failed:\n{e}")

def show_roc():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test set)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

root = tk.Tk()
root.title("Framingham Heart Disease - Logistic Regression GUI")

frm_top = ttk.Frame(root, padding=10)
frm_top.grid(row=0, column=0, sticky="ew")
lbl_title = ttk.Label(frm_top, text="Framingham Heart Disease Prediction", font=("Arial", 16, "bold"))
lbl_title.pack()

frm_metrics = ttk.Frame(root, padding=10)
frm_metrics.grid(row=1, column=0, sticky="ew")
metrics_text = (
    f"Rows: {df.shape[0]}  Columns: {df.shape[1]}\n"
    f"Target: {TARGET}\n"
    f"CV ROC AUC mean: {cv_scores.mean():.4f}\n"
    f"Test Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC AUC: {roc_auc:.4f}\n"
    f"Confusion Matrix:\n{cm}"
)
lbl_metrics = ttk.Label(frm_metrics, text=metrics_text, justify="left")
lbl_metrics.pack()

frm_inputs = ttk.Frame(root, padding=10)
frm_inputs.grid(row=2, column=0, sticky="nsew")
canvas = tk.Canvas(frm_inputs)
scrollbar = ttk.Scrollbar(frm_inputs, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set, height=300)

entries = {}
for i, f in enumerate(FEATURES):
    lbl = ttk.Label(scrollable_frame, text=f)
    lbl.grid(row=i, column=0, sticky="w", padx=5, pady=3)
    ent = ttk.Entry(scrollable_frame, width=20)
    ent.insert(0, str(medians.get(f, "")))
    ent.grid(row=i, column=1, padx=5, pady=3)
    entries[f] = ent

canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

frm_actions = ttk.Frame(root, padding=10)
frm_actions.grid(row=3, column=0, sticky="ew")
btn_predict = ttk.Button(frm_actions, text="Predict from inputs", command=predict_from_inputs)
btn_predict.grid(row=0, column=0, padx=5)
btn_save = ttk.Button(frm_actions, text="Save Model", command=save_model)
btn_save.grid(row=0, column=1, padx=5)
btn_roc = ttk.Button(frm_actions, text="Show ROC Curve", command=show_roc)
btn_roc.grid(row=0, column=2, padx=5)
lbl_result = ttk.Label(frm_actions, text="Prediction: N/A  Probability: N/A")
lbl_result.grid(row=1, column=0, columnspan=3, pady=8)

root.mainloop()
