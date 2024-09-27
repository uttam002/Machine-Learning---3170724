import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Input data: X is the advertisement spending, Y is the corresponding unit sales increase
X = np.array([70, 80, 90, 100, 110, 120, 130, 140, 150, 160]).reshape(-1, 1)
Y = np.array([7, 7, 8, 9, 12, 12, 15, 14, 13, 17])

# Create a linear regression model
model = LinearRegression()
model.fit(X, Y)

# Find the model parameters: intercept (w0) and slope (w1)
w0 = model.intercept_  # Intercept: w0 (predicted sales when spending is 0)
w1 = model.coef_[0]    # Slope: w1 (increase in sales per dollar spent)

# Predict the unit sales (Y) for X = 210
X_new = np.array([[210]])  # New advertisement spending
Y_pred = model.predict(X_new)

# Evaluate the model: MSE and R-squared (accuracy)
Y_pred_all = model.predict(X)  # Predicted sales for all given spending values
mse = mean_squared_error(Y, Y_pred_all)  # Mean Squared Error (MSE)
r2 = r2_score(Y, Y_pred_all)  # R-squared (accuracy of the model)

# Output the model parameters, prediction for X = 210, and performance metrics
output = {
    'Intercept (w0)': w0,
    'Slope (w1)': w1,
    'Predicted sales for X = 210': Y_pred[0],
    'Mean Squared Error (MSE)': mse,
    'R-squared (R2)': r2
}

# Plot the data points
plt.scatter(X, Y, color="blue", label="Data points")

# Plot the regression line
plt.plot(X, model.predict(X), color="red", label="Regression line")

# Show the predicted point for X = 210
plt.scatter(X_new, Y_pred, color="green", label=f"Predicted Y for X=210 ({Y_pred[0]:.2f})", zorder=5)

# Add labels and title
plt.xlabel("Advertisement Spending ($)")
plt.ylabel("Increase in Unit Sales")
plt.title("Linear Regression: Advertisement Spending vs Unit Sales")

# Display the legend
plt.legend()

# Show the plot
plt.show()

# output
# Intercept (w0) : -11.339285714285717    # Intercept: baseline sales when no money is spent
# Slope (w1) : 0.17678571428571427     # Slope: increase in sales per dollar spent
# Predicted sales for X = 210 : 25.845        # Prediction for $210 advertisement spending
# Mean Squared Error (MSE) : 3.7303571428571445
# R-squared (R2) : 0.9254232804232804        # Model accuracy

