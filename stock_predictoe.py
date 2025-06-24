import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = yf.download('AAPL', start='2022-01-01', end='2024-12-31')
df = df[['Close']]
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df[['Close']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Price')
plt.plot(predicted, label='Predicted Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, predicted)
print("Mean Squared Error:", mse)