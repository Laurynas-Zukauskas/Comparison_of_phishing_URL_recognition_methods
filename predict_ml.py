import pandas as pd
import time
from joblib import load

def evaluate_model(model, urls, labels):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    preds = model.predict(urls)
    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]
        if pred == 1:
            if label == 0:
                FP += 1
            else:
                TP += 1
        elif pred == 0:
            if label == 0:
                TN +=1
            else:
                FN += 1
        else:
            print("Error: prediction was", pred)
    return (TP, FP, FN, TN)

def test_models(models, dataset_file, results_file):
    test_dataset = pd.read_csv(dataset_file)
    urls = test_dataset["URL"].values.tolist()
    labels = [1 if label == "phishing" else 0 for label in test_dataset["Label"].values.tolist()]
    try:
        results = pd.read_csv(results_file)
    except FileNotFoundError:
        results = pd.DataFrame(
            {
                "model" : [],
                "TP" : [],
                "FP" : [],
                "FN" : [],
                "TN" : [],
                "accuracy" : [],
                "time" : []
            }
        )
    for model in models:
        try:
            start = time.time()
            tp, fp, fn, tn = evaluate_model(model, urls, labels)
            end = time.time()
            acc = (tp + tn) / (tp + fp + fn + tn)
            results.loc[len(results)] = {
                "model" : type(model[1]).__name__,
                "TP" : tp,
                "FP" : fp,
                "FN" : fn,
                "TN" : tn,
                "accuracy" : acc,
                "time" : end - start
            }
            results.to_csv(results_file, index=False)
        except ZeroDivisionError:
            print("divide by 0")
        except Exception as e:
            print(f"Exception: {e}\nmodel is {model}")

test_dataset_file = "test_dataset.csv"
feature_types = ["lexical", "tf-idf", "full"]

for ft in feature_types:
    models = [load(f"models/{ft}/LogisticRegression.joblib"), load(f"models/{ft}/LinearSVC.joblib"), load(f"models/{ft}/ComplementNB.joblib"), load(f"models/{ft}/KNeighborsClassifier.joblib"), load(f"models/{ft}/DecisionTreeClassifier.joblib"), load(f"models/{ft}/RandomForestClassifier.joblib"), load(f"models/{ft}/XGBClassifier.joblib")]
    test_models(models, test_dataset_file, f"ml_{ft}_results0.csv")