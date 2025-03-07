"""IDS (Anomaly Detection) + Intersection Feature Selection"""
"""Writen By: M. Aidiel Rachman Putra"""
"""Organization: Net-Centic Computing Laboratory | Institut Teknologi Sepuluh Nopember"""

import warnings
warnings.simplefilter(action='ignore')
import os
import pandas as pd

import feature_selection as featureSelection
import pre_processing as preprocessing
import classification as classification

from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    
    DATA_TRAINING_LOCATION = os.getenv('DATA_TRAINING_LOCATION')
    DATA_TESTING_LOCATION = os.getenv('DATA_TESTING_LOCATION')

    train = pd.read_csv(DATA_TRAINING_LOCATION)
    test = pd.read_csv(DATA_TESTING_LOCATION)
    k_value = int(input("Submit number of feature that will be included in intersection analysis: "))

    train, test = preprocessing.cleansing(train, test)
    X_train, X_test, y_train, y_test = preprocessing.normalization(train, test)

    pearson_features = featureSelection.pearsonCorrelation(X_train, y_train, k_value)
    kendall_features = featureSelection.kendallCorrelation(pd, X_train, y_train, k_value)
    intersection_features = featureSelection.intersection(pearson_features, kendall_features)

    model = classification.train(X_train, y_train)
    classification.test(model, X_test, y_test)



    
    