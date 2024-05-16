import pandas as pd
import os

os.environ['KERAS_BACKEND'] = 'torch'
# import keras
serieAdf = []
for file in os.listdir("data/whole/fillna"):
    if "csv" in file and "Serie-A" in file:
        print(file)
        df = pd.read_csv("./data/whole/fillna/"+file)
        serieAdf.append(df)
        print("")

serieAdf = pd.concat(serieAdf)
# Preprocess data for training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
# use one hot encoder to encode the categorical columns and standard scaler to standardize the data
x = serieAdf.drop(columns=["Player", 'value'])
y = serieAdf['value']
oeh = OneHotEncoder(sparse=False)
x = oeh.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Standardize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler.transform(y_test.values.reshape(-1, 1))
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = XGBRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))
import autosklearn.regression
cls = autosklearn.regression.AutoSklearnRegressor(memory_limit=8000)
cls.fit(x_train, y_train)
predictions = cls.predict(x_test)