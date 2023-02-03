import numpy as np
from sklearn.linear_model import LinearRegression

# Organize the data
hours = np.array([2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8])
scores = np.array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(hours.reshape(-1,1), scores)

# Predict the score for 9.25 hours of study
predicted_score = model.predict([[9.25]])
print(predicted_score)
