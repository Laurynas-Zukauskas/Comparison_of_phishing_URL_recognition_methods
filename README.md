# Comparison_of_phishing_URL_recognition_methods
This repository contains the source code produced for the experiments of the Bachelor's thesis "Comparison of phishing URL recognition methods".

features.py contains functions used for feature engineering of ML models.<br>
train_ml_models.py trains ML models using the extracted features from features.py and saves them to files.<br>
predict_ml.py is used to test the accuracy of the ML models and record the results.<br>
roc_plot.py is used to create and save ROC graphs of the ML models.

predict3.py is used to test the accuracy of selected LLM models from Hugging Face and record the results.<br>
predict_tuned.py is used to test base LLMs and compare them with fine-tuned LLMs.
