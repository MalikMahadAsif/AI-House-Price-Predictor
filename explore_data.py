
# Import the required libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Download and load the housing dataset
housing = fetch_california_housing(as_frame=True)

# Convert the data into a Pandas DataFrame
df = housing.frame

# Display the first five rows
print("First 5 rows of our dataset:")
print(df.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display the column names
print("\nColumn names:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
