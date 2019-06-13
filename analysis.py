from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spec_calculator import SpecCalculator
from helpers import find_best_val

df = pd.read_csv('dataset.csv')

msk = np.random.rand(len(df)) < 0.5

spec_c = df[msk].drop(columns=['brand', 'model', 'generation', 'hashtags'])

analyse = df[~msk].drop(columns=['brand', 'model', 'generation', 'hashtags'])

base_cols = list(df.drop(columns=['brand', 'model', 'generation', 'hashtags', 'sentiment']).columns)

# Analyser

X_an = analyse.drop(columns=['sentiment'])
y_an = analyse['sentiment']

# print (X_an, y_an)

analyser = LinearRegression().fit(X_an, y_an)

# Spec Calculator

x_points = list(range(50, 150))
y_points = []

for x in x_points:

    col = 'power'

    X_spec = spec_c[[col,'sentiment']]
    Y_spec = spec_c.drop(columns=[col, 'sentiment'])

    spec = SpecCalculator(n_columns=16, n_estimators=x)

    spec.fit(X_spec, Y_spec)

    # predicting
    base_val = find_best_val(X_spec, col)

    X = np.reshape([base_val, 1], [1,-1])

    Y = spec.predict(X)[0]

    data = {}

    data[col] = base_val

    for i,col in enumerate(Y_spec.columns):
        data[col] = [Y[i]]
    
    Y_pred = pd.DataFrame.from_dict(data)

    senti = analyser.predict(Y_pred)

    print(senti)

# for col in base_cols:

#     X_spec = spec_c[[col,'sentiment']]
#     Y_spec = spec_c.drop(columns=[col, 'sentiment'])

#     spec = SpecCalculator(n_columns=16)

#     spec.fit(X_spec, Y_spec)

#     # predicting
#     base_val = find_best_val(X_spec, col)

#     X = np.reshape([base_val, 1], [1,-1])

#     Y = spec.predict(X)[0]

#     data = {}

#     data[col] = base_val

#     for i,col in enumerate(Y_spec.columns):
#         data[col] = [Y[i]]
    
#     Y_pred = pd.DataFrame.from_dict(data)

#     senti = analyser.predict(Y_pred)

#     y_points.append(senti)

plt.xlabel('n_estimators')
plt.ylabel('score')

plt.plot(x_points, y_points)
plt.show()
