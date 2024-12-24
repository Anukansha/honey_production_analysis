# Import necessary libraries
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For visualization
import numpy as np  # For numerical operations
from sklearn.linear_model import LinearRegression  # For linear regression model

# Read the dataset from a CSV file
df = pd.read_csv("data/honeyproduction.csv")

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Group data by "totalprod" and calculate the mean for each year
prod_per_year = df.groupby(by="year").mean().reset_index()

# Display the grouped data to check the structure
print(prod_per_year)

# Extract "year" column as the independent variable (X)
X = prod_per_year["year"].values.reshape(-1, 1)  # Reshape into 2D array for sklearn

# Extract "totalprod" column as the dependent variable (y)
y = prod_per_year["totalprod"]

# Scatter plot to visualize the data
plt.scatter(X, y)
plt.xlabel("Year")
plt.ylabel("Total Production")
plt.title("Honey Production Over Time")
plt.show()

# Create and fit a linear regression model
regr = LinearRegression().fit(X, y)

# Print model coefficients
print("Slope (Coefficient):", regr.coef_[0])
print("Intercept:", regr.intercept_)

# Predict production for the given years
y_predict = regr.predict(X)

# Plot original data and the regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_predict, color="red", label="Regression Line")
plt.xlabel("Year")
plt.ylabel("Total Production")
plt.title("Honey Production and Regression Line")
plt.legend()
plt.show()

# Predict future honey production from 2013 to 2050
X_future = np.array(range(2013, 2051)).reshape(-1, 1)
future_predict = regr.predict(X_future)

# Plot the future predictions
plt.plot(X_future, future_predict, color="green", label="Future Predictions")
plt.xlabel("Year")
plt.ylabel("Total Production")
plt.title("Honey Production Prediction (2013-2050)")
plt.legend()
plt.show()

