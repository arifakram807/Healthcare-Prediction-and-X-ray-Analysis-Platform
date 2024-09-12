import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/LengthOfStay.csv')

# Convert dates to datetime format
data['vdate'] = pd.to_datetime(data['vdate'], format='%m/%d/%Y')
data['discharged'] = pd.to_datetime(data['discharged'], format='%m/%d/%Y')

# Time Series Features: Extract visit month and day of the week
data['visit_month'] = data['vdate'].dt.month
data['visit_dayofweek'] = data['vdate'].dt.dayofweek

# Replace '5+' with 5 in 'rcount' and convert to integer
data['rcount'] = data['rcount'].replace('5+', 5).astype(int)

# Sum binary health condition flags into a 'total_issues' feature
health_flags = ['dialysisrenalendstage', 'asthma', 'irondef', 'pneum',
                'substancedependence', 'psychologicaldisordermajor',
                'depress', 'psychother', 'fibrosisandother', 'malnutrition',
                'hemo']
data['total_issues'] = data[health_flags].sum(axis=1)

# Feature Engineering: Interaction Terms
data['bmi_glucose'] = data['bmi'] * data['glucose']
data['bmi_creatinine'] = data['bmi'] * data['creatinine']

# Health Risk Score (assign higher weights to more serious conditions)
health_risk_weights = {
    'dialysisrenalendstage': 3,
    'asthma': 1,
    'irondef': 1,
    'pneum': 2,
    'substancedependence': 2,
    'psychologicaldisordermajor': 2,
    'depress': 1,
    'psychother': 1,
    'fibrosisandother': 2,
    'malnutrition': 2,
    'hemo': 1
}
# Create a weighted health risk score
data['health_risk_score'] = data[health_flags].apply(lambda row: sum(health_risk_weights[flag] * row[flag] for flag in health_flags), axis=1)

# Drop unnecessary columns
X = data.drop(columns=['lengthofstay', 'vdate', 'discharged', 'eid', 'facid'])
y = data['lengthofstay']

# One-hot encode categorical features like 'gender' and 'secondarydiagnosisnonicd9'
X = pd.get_dummies(X, columns=['gender', 'secondarydiagnosisnonicd9'], drop_first=False)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the column names to ensure proper input structure during prediction
with open('models/columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the model's performance
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Print some example predictions and compare with actual values
for i in range(5):
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")

# ---------- Clustering (K-Means) ----------
# Clustering with K-Means: Use clinical features to group patients
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

# Save clustered data for analysis
data.to_csv('data/clustered_data.csv', index=False)

# Visualize the clusters based on BMI and glucose levels
plt.figure(figsize=(10, 6))
plt.scatter(data['bmi'], data['glucose'], c=data['cluster'], cmap='viridis')
plt.title('K-Means Clustering of Patients based on BMI and Glucose')
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.colorbar(label='Cluster')
plt.savefig('static/cluster_plot.png')
plt.show()

# -----------------------------------------
# Feature Importance Visualization
feature_importance = model.feature_importances_

# Create a dataframe for better visualization
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in RandomForest Model with Feature Engineering')
plt.gca().invert_yaxis()  # Invert y-axis to show the highest importance on top
plt.show()
