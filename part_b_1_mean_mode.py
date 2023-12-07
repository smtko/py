# statistical_measures.py

import numpy as np
import matplotlib.pyplot as plt

# Function 1: Generate Example Data
def generate_example_data():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(scale=2, size=len(x))
    return x, y



# Function 2: Calculate Mean
def calculate_mean(data):
    return sum(data) / len(data)

# Function 3: Calculate Variance
def calculate_variance(data, mean):
    return sum((xi - mean) ** 2 for xi in data) / len(data)

# Function 4: Calculate Standard Deviation
def calculate_std_deviation(variance):
    return np.sqrt(variance)

# Function 5: Calculate Covariance
def calculate_covariance(x, y, mean_x, mean_y):
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)
    return cov

# Function 6: Calculate Correlation
def calculate_correlation(covariance, std_dev_x, std_dev_y):
    return covariance / (std_dev_x * std_dev_y)

# Function 7: Calculate Standard Error of the Mean
def calculate_standard_error(std_dev, sample_size):
    return std_dev / np.sqrt(sample_size)

# Function 8: Display Scatter Plot
def display_scatter_plot(x, y):
    plt.scatter(x, y, label='Data Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot')
    plt.legend()
    plt.show()

# Function 9: Display Regression Line
def display_regression_line(x, y, slope, intercept):
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, slope * x + intercept, color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regression Line')
    plt.legend()
    plt.show()

# Generate Example Data
x, y = generate_example_data()

# Display Scatter Plot
display_scatter_plot(x, y)

# Function 10: Calculate Mean of X and Y
mean_x = calculate_mean(x)
mean_y = calculate_mean(y)

# Function 11: Calculate Variance of X and Y
variance_x = calculate_variance(x, mean_x)
variance_y = calculate_variance(y, mean_y)

# Function 12: Calculate Standard Deviation of X and Y
std_dev_x = calculate_std_deviation(variance_x)
std_dev_y = calculate_std_deviation(variance_y)

# Function 13: Calculate Covariance between X and Y
covariance_xy = calculate_covariance(x, y, mean_x, mean_y)

# Function 14: Calculate Correlation between X and Y
correlation_xy = calculate_correlation(covariance_xy, std_dev_x, std_dev_y)

# Function 15: Calculate Standard Error of the Mean
sample_size = len(x)
standard_error_x = calculate_standard_error(std_dev_x, sample_size)
standard_error_y = calculate_standard_error(std_dev_y, sample_size)

# Display Regression Line
slope = covariance_xy / variance_x
intercept = mean_y - slope * mean_x
display_regression_line(x, y, slope, intercept)


print("Mean of X:", mean_x)
print("Mean of Y:", mean_y)
print("Variance of X:", variance_x)
print("Variance of Y:", variance_y)
print("Standard Deviation of X:", std_dev_x)
print("Standard Deviation of Y:", std_dev_y)
print("Covariance between X and Y:", covariance_xy)
print("Correlation between X and Y:", correlation_xy)
print("Standard Error of the Mean for X:", standard_error_x)
print("Standard Error of the Mean for Y:", standard_error_y)
