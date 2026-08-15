import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
print("---Load Dataset---") 
data = pd.read_csv("E:\\DATA SCIENCE\\ML\\WEEK1 ML\\Advertising.csv") 
print("Dataset Preview:") 
print(data.head()) 
print("\nDataset Info:") 
print(data.info()) 
print("---Preprocessing---") 
X = data[['TV']]     
# Feature 
y = data['sales']    # Target 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size=0.2, random_state=42 
) 
print("---Simple Linear Regression using sklearn---") 
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred_sklearn = model.predict(X_test) 
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn) 
print("\n--- Sklearn Linear Regression ---") 
print("Slope (m):", model.coef_[0]) 
print("Intercept (c):", model.intercept_) 
print("MSE:", mse_sklearn) 
print("---Manual Linear Regression---") 
x = X_train.values.flatten() 
y_manual = y_train.values 
n = len(x) 
sum_x = np.sum(x) 
sum_y = np.sum(y_manual) 
sum_xy = np.sum(x * y_manual) 
sum_x2 = np.sum(x ** 2) 
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) 
c = (sum_y - m * sum_x) / n 
print("---Manual Predictions---") 
x_test = X_test.values.flatten() 
y_pred_manual = m * x_test + c 
mse_manual = mean_squared_error(y_test, y_pred_manual) 
print("\n--- Manual Linear Regression ---") 
print("Slope (m):", m) 
print("Intercept (c):", c) 
print("MSE:", mse_manual) 
print("---Comparison Table---") 
comparison = pd.DataFrame({ 
"TV Budget": x_test, 
"Actual Sales": y_test.values, 
"Sklearn Prediction": y_pred_sklearn, 
"Manual Prediction": y_pred_manual 
}) 
print("\nComparison Table:") 
print(comparison) 
print("---Export Results---") 
comparison.to_csv("advertising_predictions.csv", index=False) 
print("\nPredictions saved to advertising_predictions.csv")