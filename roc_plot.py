import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from joblib import load

dataset = pd.read_csv("test_dataset.csv")
urls = dataset["URL"].values.tolist()
labels = [1 if label == "phishing" else 0 for label in dataset["Label"].values.tolist()]

feature_types = [("Lexical", "lexical"), ("Character n-gram", "tf-idf"), ("Lexical + Character n-gram", "full")]
classifiers = [("Logistic Regression", "LogisticRegression"), ("Linear Support Vector Machine", "LinearSVC"), ("Naïve Bayes", "ComplementNB"), ("K-Nearest Neighbors", "KNeighborsClassifier"), ("Decision Tree", "DecisionTreeClassifier"), ("Random Forest", "RandomForestClassifier"), ("Extreme Gradient Boosting", "XGBClassifier")]
 
for c_title, c in classifiers:
    plt.figure(figsize=(7, 7)) 
    models = [load(f"models/{ft}/{c}.joblib") for _, ft in feature_types]
    for f_title, f_type in feature_types:
        model = load(f"models/{f_type}/{c}.joblib")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(urls)[:, 1]
        else:
            probs = model.decision_function(urls)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{f_title} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {c_title}')
    plt.legend()
    plt.savefig(f"{c}_roc.png")