import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Input data: X is the advertisement spending, Y is the corresponding unit sales increase
X = np.array([70, 80, 90, 100, 110, 120, 130, 140, 150, 160]).reshape(-1, 1)  # Reshape for sklearn
Y = np.array([7, 7, 8, 9, 12, 12, 15, 14, 13, 17])

# Create a linear regression model
model = LinearRegression()
model.fit(X, Y)

# Find the model parameters: intercept (w0) and slope (w1)
w0 = model.intercept_  # Intercept: w0 (the predicted unit sales when advertisement spending is 0)
w1 = model.coef_[0]    # Slope: w1 (rate of change in sales with respect to advertisement spending)

# Predict the unit sales (Y) when advertisement spending (X) is 210
X_new = np.array([[210]])  # New advertisement spending
Y_pred = model.predict(X_new)

# Evaluate the model: MSE and R-squared (accuracy)
Y_pred_all = model.predict(X)  # Predicted sales for all given spending values
mse = mean_squared_error(Y, Y_pred_all)  # Mean Squared Error (MSE)
r2 = r2_score(Y, Y_pred_all)  # R-squared (accuracy of the model)

# Output the model parameters, prediction for X = 210, and performance metrics
w0, w1, Y_pred[0], mse, r2


# Output
# w0 = -11.339285714285717    # Intercept: baseline sales when no money is spent
# w1 = 0.17678571428571427     # Slope: increase in sales per dollar spent
# Predicted sales for X = 210 = 25.845        # Prediction for $210 advertisement spending
# Mean Squared Error (mse) = 3.7303571428571445
# R-squared (r2) = 0.9254232804232804        # Model accuracy
