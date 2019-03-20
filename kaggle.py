import csv
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split

def process_file(filename):
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',')
    data = list(reader)
    data = data[:][1:]
    return np.array(data)

X_train = pd.read_csv('data/X_train.csv', header=0).as_matrix()
y_train = pd.read_csv('data/y_train.csv', header=0).as_matrix()
X_test = pd.read_csv('data/X_test.csv', header=0).as_matrix()

model = XGBRegressor(max_depth=3)
n_estimators = [400]
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)
print(grid_result.best_params_)

y_test = grid_result.predict(X_test)

predictions = open('predictions4.csv', 'w')
writer = csv.writer(predictions)
writer.writerow(["id", "actual_wait div 60000"])
id = 0
for i in range(0, len(y_test)):
    if (y_test[i] < 0):
        y_test[i] = 0
    entry = [id, y_test[i]]
    writer.writerow(entry)
    id += 1