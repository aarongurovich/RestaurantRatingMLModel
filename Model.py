import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
businesses = pd.read_json('yelp_business.json', lines=True)
checkins = pd.read_json('yelp_checkin.json', lines=True)
photos = pd.read_json('yelp_photo.json', lines=True)
reviews = pd.read_json('yelp_review.json', lines=True)
users = pd.read_json('yelp_user.json', lines=True)
tips = pd.read_json('yelp_tip.json', lines=True)

# Display settings
pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500

# Merge datasets
df = pd.merge(businesses, reviews, how='left', on='business_id')
df = pd.merge(df, users, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')

# Drop unnecessary columns
features_to_remove = ['address', 'attributes', 'business_id', 'categories', 'city', 'hours', 'is_open', 'latitude', 'longitude', 'name', 'neighborhood', 'postal_code', 'state', 'time']
df.drop(features_to_remove, axis=1, inplace=True, errors='ignore')

# Fill missing values
df.fillna(0, inplace=True)

correlation_matrix = df.corr()
high_correlation_features = correlation_matrix.index[abs(correlation_matrix['stars']) > 0.1].tolist()  # Adjust the threshold as needed

# Remove 'stars' from the list of features
high_correlation_features.remove('stars')

# Define features (X) and target (y)
X = df[high_correlation_features]
y = df['stars']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Clip predictions to ensure they start at 1
y_pred = y_pred.clip(min=1, max=5)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting true values vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('True Values vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# Test a single data point
single_data_point = X_test.iloc[[0]]  # Select the first data point in the test set
true_value = y_test.iloc[0]  # Get the true value of the first data point in the test set

# Make a prediction for the single data point
single_prediction = model.predict(single_data_point)

# Clip the prediction to ensure it is between 1 and 5
single_prediction = single_prediction.clip(min=1, max=5)

print(f'True value: {true_value}')
print(f'Predicted value: {single_prediction[0]}')

# Plotting the single data point prediction
plt.figure(figsize=(6, 6))
plt.scatter(true_value, single_prediction, color='red', marker='o', s=100)
plt.plot([1, 5], [1, 5], '--r', linewidth=2)
plt.title('True Value vs Predicted Value for a Single Data Point')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.grid(True)
plt.show()
