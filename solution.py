import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data_from_file = pd.read_csv('salaries.csv')

data_x = data_from_file.iloc[:, 2:5]
data_y = data_from_file.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=0)

# ========================== Linear Regression ================================
lin_reg = LinearRegression()
# lin_reg.fit(x_train, y_train)

lin_reg.fit(X = x_train, y = y_train)
y_linear_predict1 = lin_reg.predict(x_test)

# Performans değerlendirme (R2 ve MSE)
mse1 = mean_squared_error(y_test, y_linear_predict1)
r2_1 = r2_score(y_test, y_linear_predict1)
# print(f"Original Linear Regression Model - MSE: {mse1}, R²: {r2_1}")

x_train = x_train.reset_index(drop=True)

# ========================== OLS (Linear Model) ===============================
bias_linear = pd.DataFrame(np.ones((x_train.shape[0], 1)).astype(int), columns=['bias']).reset_index(drop=True)

final_train_data_with_bias_linear = pd.concat([bias_linear, x_train], axis=1, ignore_index=False)

y_train = y_train.reset_index(drop = True)

X_l = final_train_data_with_bias_linear.iloc[:, [0,1,2]].values
regressor_OLS = sm.OLS(endog = y_train, exog = X_l)
# print("Linear result:")
# print(regressor_OLS.fit().summary())

lin_reg_bias = LinearRegression()
lin_reg_bias.fit(X = X_l, y = y_train)
x_test_linear_bias = sm.add_constant(x_test)
y_linear_predict2 = lin_reg_bias.predict(x_test_linear_bias.iloc[:, [0,1,2]])

# Performans değerlendirme (R2 ve MSE)
mse2 = mean_squared_error(y_test, y_linear_predict2)
r2_2 = r2_score(y_test, y_linear_predict2)
# print(f"After Feature Removal Linear Regression Model - MSE: {mse2}, R²: {r2_2}")

# ========================== Polynomial Regression ============================
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x_train)
# print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(X = x_poly,y = y_train)

x_poly_test = poly_reg.transform(x_test)
y_poly_predict1 = lin_reg2.predict(x_poly_test)

mse_poly1 = mean_squared_error(y_test, y_poly_predict1)
r2_poly1 = r2_score(y_test, y_poly_predict1)
# print(f"Original Poly Regression Model - MSE: {mse_poly1}, R²: {r2_poly1}")

# ========================== OLS (Polynomial Model) ===========================

regressor_OLS_poly = sm.OLS(endog=y_train, exog=x_poly)
result = regressor_OLS_poly.fit()

final_train_data_with_bias_poly = x_poly.copy()

p_values = result.pvalues
high_p_value_indices = np.where(p_values > 0.5)[0]
filtered_data_x = np.delete(final_train_data_with_bias_poly, high_p_value_indices, axis=1)

x_test_poly_copy = np.delete(x_poly_test, high_p_value_indices, axis=1)

regressor_OLS_filtered = sm.OLS(endog=y_train, exog=filtered_data_x)
result_filtered = regressor_OLS_filtered.fit()

# print("Poly Result:")
# print(result_filtered.summary())

lin_reg2.fit(X = filtered_data_x[:, :], y = y_train)
y_poly_predict2 = lin_reg2.predict(x_test_poly_copy)

mse_poly2 = mean_squared_error(y_test, y_poly_predict2)
r2_poly2 = r2_score(y_test, y_poly_predict2)
# print(f"After Feature Removal Poly Regression Model - MSE: {mse_poly2}, R²: {r2_poly2}")

# print(f"predict1: {y_poly_predict1}\n predict2: {y_poly_predict1}\n y_test: {y_test}")

# ========================== SVR (Support Vector Regression) ==================

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

y_train_scaled = scaler_y.fit_transform(y_train)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_train_scaled, y_train_scaled.ravel())

# ========================== OLS (SVR Model) ==================================
x_train_ols = x_train.copy()
ols_model = sm.OLS(y_train, x_train_ols).fit()

p_values = ols_model.pvalues

high_p_value_indices = p_values[p_values > 0.5].index 

high_p_value_indices = high_p_value_indices.drop('const', errors='ignore')

x_train_filtered = x_train.drop(columns=high_p_value_indices)
x_test_filtered = x_test.drop(columns=high_p_value_indices)

regressor_OLS_filtered = sm.OLS(endog=y_train, exog=x_train_filtered)
result_filtered = regressor_OLS_filtered.fit()

print("SVR Result:")
print(result_filtered.summary())

# ========================== Decision Tree ====================================

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(data_x,data_y)

print('Decision Tree OLS')
dt_ols = sm.OLS(r_dt.predict(data_x),data_x)
print(dt_ols.fit().summary())

print('Decision Tree R2 degeri')
print(r2_score(data_y, r_dt.predict(data_x)))

# ========================== Random Forest ====================================
rf_reg=RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(data_x, data_y.values.ravel())

print('Random Forest OLS')
rf_ols = sm.OLS(rf_reg.predict(data_x),data_x)
print(rf_ols.fit().summary())


print('Random Forest R2 degeri')
print(r2_score(data_y, rf_reg.predict(data_x)))