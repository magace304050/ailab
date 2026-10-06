import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

print("=" * 60)
print("   STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 60)

print("\nDataset Information")
print("-" * 60)
print("Total Students :", len(data))
print("Total Columns  :", len(data.columns))

# 2. Select real UCI features
features = ["G1", "G2", "studytime", "absences"]
target = "G3"

X = data[features]
y = data[target]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=4,
    random_state=42
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Test model
y_pred = model.predict(X_test)

# 7. Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Information")
print("-" * 60)
print("Algorithm       : Random Forest Regressor")
print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

print("\nModel Performance")
print("-" * 60)
print(f"MAE             : {mae:.3f}")
print(f"RMSE            : {rmse:.3f}")
print(f"R2 Score        : {r2:.4f}")
print(f"R2 Percentage   : {r2 * 100:.2f}%")

# 8. Predict a student
print("\nStudent Prediction")
print("-" * 60)

g1 = float(input("Enter First Period Mark (G1) [0-20]: "))
g2 = float(input("Enter Second Period Mark (G2) [0-20]: "))
studytime = float(input("Enter Study Time Category [1-4]: "))
absences = float(input("Enter Number of Absences: "))

student = pd.DataFrame(
    [[g1, g2, studytime, absences]],
    columns=features
)

prediction = model.predict(student)[0]

# Keep prediction within valid UCI grade range
prediction = max(0, min(20, prediction))

print("\nPrediction Result")
print("-" * 60)
print(f"Predicted Final Mark (G3): {prediction:.2f} / 20")

if prediction >= 15:
    category = "Excellent"
elif prediction >= 12:
    category = "Good"
elif prediction >= 10:
    category = "Average"
else:
    category = "Needs Improvement"

print("Performance Category      :", category)

print("\n" + "=" * 60)
print("Prediction completed successfully.")
print("=" * 60)