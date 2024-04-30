import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\micha\OneDrive\Documents\hello\Hackathon_Working_Data.csv')

# Data Cleaning
# Handle missing values
df = df.dropna()


# Remove duplicates
df = df.drop_duplicates()

# Descriptive Statistics
print(df.describe())

# Data Visualization
# Histograms for distribution
df.hist(figsize=(10, 8))
df = pd.DataFrame(df)
# Box plots for outliers
sns.boxplot(data=df)



# Trend Analysis
# Line chart for sales trends
plt.figure(figsize=(10, 5))
plt.plot(df['STORECODE'], df['QTY'])
plt.title('Sales Trends Over Time')
plt.xlabel('STORECODE')
plt.ylabel('QTY')
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(df['MONTH'], df['QTY'])
plt.title('Sales Trends Over Time')
plt.xlabel('MONTH')
plt.ylabel('QTY')
plt.show()

# Customer Segmentation (Example using K-Means Clustering)
from sklearn.cluster import KMeans

# Assuming 'features_to_cluster' is a list of column names you want to use for clustering
X = df[['QTY','VALUE']].values
kmeans = KMeans(n_clusters=2)
df['Cluster'] = kmeans.fit_predict(X)
y=df['BILL_AMT']
# Predictive Analytics (Example using Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Assuming 'X' is your set of features and 'y' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
