import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Nairobi Office Price Ex.csv")
print(data.head())

# Feature and target values
x = data['SIZE'].values
y = data['PRICE'].values

# Min-max scaling for both x and y
x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))

# Define the mean squared error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(x_scaled, y_scaled, learning_rate=0.1, epochs=10):
    m = 0.0  # Initialize slope
    c = 0.0  # Initialize intercept
    n = len(y_scaled)  # Number of data points

    for epoch in range(epochs):
        # Predict the y values (y = mx + c)
        y_pred = (m * x_scaled) + c

        # Calculate gradients
        dm = (-2 / n) * sum(x_scaled * (y_scaled - y_pred))
        dc = (-2 / n) * sum(y_scaled - y_pred)

        # Update m and c
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Optionally, print error to track progress
        if epoch % 1 == 0:
            error = mean_squared_error(y_scaled, y_pred)
            print(f"Epoch {epoch + 1}: MSE = {error}")

    return m, c

# Call gradient descent with scaled data
m, c = gradient_descent(x_scaled, y_scaled)

# Generate predictions for plotting
y_pred = m * x_scaled + c

# Plot the actual data and the line of best fit
plt.scatter(x, y, color='green', label='Actual data')
plt.plot(x, y_pred * (np.max(y) - np.min(y)) + np.min(y), color='red', label='Line of best fit')  # Reverse scaling for y_pred
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.legend()
plt.show()

# Predict for an office of 100 sq. ft. (scaled value)
office_size = 100
office_size_scaled = (office_size - np.min(x)) / (np.max(x) - np.min(x))
predicted_price_scaled = m * office_size_scaled + c
predicted_price = predicted_price_scaled * (np.max(y) - np.min(y)) + np.min(y)  # Reverse scaling to get actual price
print(f"The predicted price for an office of 100 sq. ft. is {predicted_price}")
