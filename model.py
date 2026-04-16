import pandas as pd

data = pd.read_csv("ecommerce_sales_data.csv")

print(data.head())
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])
data['Days'] = (data['Purchase Date'] - data['Purchase Date'].min()).dt.days
data['Total Sales'] = data['Price'] * data['Quantity']
data = data.groupby('Days')['Total Sales'].sum().reset_index()
from sklearn.linear_model import LinearRegression

X = data[['Days']]
y = data['Total Sales']

model = LinearRegression()
model.fit(X, y)
import pickle

pickle.dump(model, open("model.pkl", "wb"))
future_day = [[100]]
prediction = model.predict(future_day)

print("Future Sales:", prediction)