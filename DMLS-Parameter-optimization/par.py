import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("ss316l.csv")

cols_to_use = ['Layer_Thickness_microns','Laser_Power_W','Scan_Speed_mm_s','Hatch_Spacing_mm',
               'Powder_Bed_Temperature_C','Material_Density_g_cm3','Quality_Rating']
X = df[cols_to_use]
y = df['Porosity_percent']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

models = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression (degree=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(),
    'k-Nearest Neighbors': KNeighborsRegressor(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\n===============================")
print("MODEL COMPARISON")
print("===============================")
print(results_df)

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_model.fit(X_scaled, y)

print("\nBest Model Selected:", best_model_name)

valid_thickness = [20, 40, 70, 90]
while True:
    layer_thickness = int(
        input(f"\nChoose Layer Thickness {valid_thickness} (microns): ")
    )
    if layer_thickness in valid_thickness:
        break
    print("Invalid input. Try again.")

param_ranges = {
    'Layer_Thickness_microns': [layer_thickness],
    'Laser_Power_W': (200, 400),
    'Scan_Speed_mm_s': (400, 1200),
    'Hatch_Spacing_mm': (0.08, 0.15),
    'Powder_Bed_Temperature_C': (150, 250),
    'Material_Density_g_cm3': (7.8, 8.1),
    'Quality_Rating': (1, 5)
}

n_samples = 10000
random_params = {}

for key, val in param_ranges.items():
    if isinstance(val, list):
        random_params[key] = np.repeat(val[0], n_samples)
    else:
        random_params[key] = np.random.uniform(val[0], val[1], n_samples)

random_df = pd.DataFrame(random_params)
random_df['Quality_Rating'] = random_df['Quality_Rating'].round().astype(int)

random_scaled = scaler.transform(random_df)
random_df['Predicted_Porosity'] = best_model.predict(random_scaled)

random_df = random_df[random_df['Predicted_Porosity'] >= 0]

best_params = random_df.loc[random_df['Predicted_Porosity'].idxmin()]

print("\n\n")
print("OPTIMAL PROCESS PARAMETERS (Predicted)")
print("\n\n")
print(f"Layer Thickness: {layer_thickness} microns")
print(f"Best Model Used: {best_model_name}")
print("\n\n")
print(best_params)
