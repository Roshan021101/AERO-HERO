WAP to implement a linear regression model for regression task and train the model
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([
    [1000, 2], [1500, 3], [1200, 2],
    [1800, 4], [900, 2], [2000, 3]
])

y = np.array([300000, 400000, 350000, 500000, 280000, 450000])

model = LinearRegression().fit(x, y)

size = float(input("Enter the size of the house in sqft: "))
bedrooms = int(input("Enter the number of bedrooms: "))

price = model.predict([[size, bedrooms]])[0]

print(f"Predicted price for a house with size {size} sqft and {bedrooms} bedrooms: Rs {price:.2f}")
