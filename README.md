# House-Price-Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic dataset
np.random.seed(42)
n_samples = 5000

data = {
    'Price': np.random.randint(50000, 500000, n_samples),
    'Area': np.random.randint(600, 5000, n_samples),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Bathrooms': np.random.randint(1, 4, n_samples),
    'Stories': np.random.randint(1, 4, n_samples),
    'Mainroad': np.random.choice(['Yes', 'No'], n_samples),
    'Guestroom': np.random.choice(['Yes', 'No'], n_samples),
    'Basement': np.random.choice(['Yes', 'No'], n_samples),
    'Hot_water_heating': np.random.choice(['Yes', 'No'], n_samples),
    'Airconditioning': np.random.choice(['Yes', 'No'], n_samples),
    'Parking': np.random.randint(0, 4, n_samples),
    'Prefarea': np.random.choice(['Yes', 'No'], n_samples),
    'Furnishing_status': np.random.choice(['Fully Furnished', 'Semi-Furnished', 'Unfurnished'], n_samples)
}

df = pd.DataFrame(data)

# Convert categorical columns to numerical
categorical_cols = ['Mainroad', 'Guestroom', 'Basement', 'Hot_water_heating', 'Airconditioning', 'Prefarea', 'Furnishing_status']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save dataset
df.to_csv("house_prices.csv", index=False)

# Display basic statistics
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Distribution of house prices
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Area vs. Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Area'], y=df['Price'])
plt.title('Area vs. Price')
plt.xlabel('Area (sqft)')
plt.ylabel('Price')
plt.show()

# Boxplot: Prices by furnishing status
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Furnishing_status_Fully Furnished'], y=df['Price'])
plt.xticks(ticks=[0, 1], labels=['Semi/Unfurnished', 'Fully Furnished'])
plt.title('House Prices by Furnishing Status')
plt.xlabel('Furnishing Status')
plt.ylabel('Price')
plt.show()
