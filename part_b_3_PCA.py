# data_correlation_pca.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

from scikit import pca, iris


# Function 1: Load Iris Dataset
def load_dataset():
    iris = load_iris()
    data = iris.data
    feature_names = iris.feature_names
    target_names = iris.target_names
    return data, feature_names, target_names


# Function 2: Create DataFrame from Iris Dataset
def create_dataframe(data, feature_names, target_names):
    columns = feature_names + ['target']
    df = pd.DataFrame(data=np.column_stack([data, iris.target]), columns=columns)
    return df


# Function 3: Display DataFrame Head
def display_dataframe_head(df):
    print("DataFrame Head:")
    print(df.head())


# Function 4: Display Data Correlation Heatmap
def display_correlation_heatmap(df):
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Data Correlation Heatmap")
    plt.show()


# Function 5: Standardize Data
def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data


# Function 6: Perform PCA
def perform_pca(standardized_data, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(standardized_data)
    return principal_components


# Function 7: Display Explained Variance
def display_explained_variance(pca):
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance:")
    print(explained_variance)


# Function 8: Display Scree Plot
def display_scree_plot(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.show()


# Function 9: Display 2D Scatter Plot of Principal Components
def display_2d_scatter_plot(principal_components, target_names):
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=iris.target, cmap='viridis', edgecolor='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D Scatter Plot of Principal Components')
    plt.show()





# Load Iris Dataset
data, feature_names, target_names = load_dataset()

# Create DataFrame
df = create_dataframe(data, feature_names, target_names)

# Display DataFrame Head
display_dataframe_head(df)

# Display Data Correlation Heatmap
display_correlation_heatmap(df)

# Select Features and Target
X = df[feature_names]
y = df['target']

# Standardize Data
standardized_data = standardize_data(X)

# Perform PCA
n_components = 2
principal_components = perform_pca(standardized_data, n_components)

# Display Explained Variance
display_explained_variance(pca)

# Display Scree Plot
display_scree_plot(pca)

# Display 2D Scatter Plot of Principal Components
display_2d_scatter_plot(principal_components, target_names)


