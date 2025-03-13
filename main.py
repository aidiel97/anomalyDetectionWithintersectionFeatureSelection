"""IDS (Anomaly Detection) + Intersection Feature Selection"""
"""Writen By: M. Aidiel Rachman Putra"""
"""Organization: Net-Centic Computing Laboratory | Institut Teknologi Sepuluh Nopember"""

import warnings
warnings.simplefilter(action='ignore')
import os
import csv
import pandas as pd

import feature_selection as featureSelection
import pre_processing as preprocessing
import classification as classification

from dotenv import load_dotenv
from datetime import datetime

if __name__ == "__main__":
    now = datetime.now()
    load_dotenv()
    
    DATA_TRAINING_LOCATION = os.getenv('DATA_TRAINING_LOCATION')
    DATA_TESTING_LOCATION = os.getenv('DATA_TESTING_LOCATION')
    OUT_DIR = os.getenv('OUT_DIR')

    train = pd.read_csv(DATA_TRAINING_LOCATION)
    test = pd.read_csv(DATA_TESTING_LOCATION)
    k_value = int(input("Submit number of feature that will be included in intersection analysis: "))
    selected_algorithm = classification.menu()

    train, test = preprocessing.cleansing(train, test)
    X_train, X_test, y_train, y_test = preprocessing.normalization(train, test)

    pearson_features = featureSelection.pearsonCorrelation(X_train, y_train, k_value)
    kendall_features = featureSelection.kendallCorrelation(pd, X_train, y_train, k_value)
    intersection_features = featureSelection.intersection(pearson_features, kendall_features)

    model = classification.train(X_train, y_train, selected_algorithm)
    tp, tn, fp, fn, accuracy, precision, recall, f1 = classification.test(model, X_test, y_test)
    
    dict = {
        "CreatedAt": now,
        "Algorithm": selected_algorithm,
        "ClassificationContext": "Intersection of "+str(k_value)+" features on Pearson and Kendall",
        "PearsonFeatures": str(pearson_features),
        "KendallFeatures": str(kendall_features),
        "SelectedFeatures": str(intersection_features),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }

    logFilePath = OUT_DIR+str(int(now.timestamp()))+'_log.txt'
    outputFilePath = OUT_DIR+'classification_results.csv'
    file_exists = os.path.exists(outputFilePath) and os.path.getsize(outputFilePath) > 0
    field_names = ['CreatedAt', 'Algorithm', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ClassificationContext', 'PearsonFeatures', 'KendallFeatures', 'SelectedFeatures']
    with open(outputFilePath, 'a', newline='') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
        
        if not file_exists:
            dict_object.writeheader()

        dict_object.writerow(dict)