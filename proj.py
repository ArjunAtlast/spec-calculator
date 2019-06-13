import pandas as pd
from helpers import find_avg_sentiment, feature_list, find_best_val
from spec_calculator import SpecCalculator
import numpy as np
from gui import GUI

df = pd.read_csv("dataset.csv")

df_vals = df.drop(columns=['brand', 'model', 'generation', 'hashtags'])

def update_sentiment():
    """
    Update the sentiment value in dataset
    """

    global df

    df['sentiment'] = df.apply(lambda row: find_avg_sentiment(row), axis=1)

    df.to_csv('dataset.csv')

def calculate(base_class):

    # create model
    sc = SpecCalculator(16, 85)

    # fit model
    X = df_vals[[base_class, 'sentiment']]
    Y = df_vals.drop(columns=[base_class, 'sentiment'])

    sc.fit(np.reshape(X.values, [-1,2]), Y)

    best_val = find_best_val(X, base_class) 

    X_pred = np.reshape([best_val, 1], [1,-1])

    Y_pred = sc.predict(X_pred)[0]

    data = {}

    data[base_class] = best_val

    for i,col in enumerate(Y.columns):
        data[col] = Y_pred[i]
    
    return data

gui = GUI(calculate, update_sentiment, feature_list())

gui.load()