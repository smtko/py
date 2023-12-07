import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# Function 1: Generate Normal Distribution
def generate_normal_distribution(mean, std_dev, size):
    return np.random.normal(mean, std_dev, size)


# Function 2: Plot Histogram
def plot_histogram(data, bins=30):
    plt.hist(data, bins=bins, density=True, alpha=0.7, color='blue')
    plt.title("Class Test Results - Normal Distribution")
    plt.xlabel("Test Scores")
    plt.ylabel("Probability Density")
    plt.show()


# Function 3: Plot Probability Density Function (PDF)
def plot_pdf(data, bins=30):
    count, bins, _ = plt.hist(data, bins=bins, density=True, alpha=0.7, color='blue')
    mu, sigma = np.mean(data), np.std(data)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='red')
    plt.title("Class Test Results - Normal Distribution with PDF")
    plt.xlabel("Test Scores")
    plt.ylabel("Probability Density")
    plt.show()


# Function 4: Plot Cumulative Distribution Function (CDF)
def plot_cdf(data):
    count, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.7, color='blue', cumulative=True, histtype='step')
    plt.title("Class Test Results - Cumulative Distribution Function (CDF)")
    plt.xlabel("Test Scores")
    plt.ylabel("Cumulative Probability")
    plt.show()


# Function 5: Calculate Skewness
def calculate_skewness(data):
    return skew(data)


# Function 6: Calculate Kurtosis
def calculate_kurtosis(data):
    return kurtosis(data)


# Function 7: Print Skewness and Kurtosis
def print_statistics(skewness, kurtosis):
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")


# Function 8: Identify Skewness
def identify_skewness(skewness):
    if skewness > 0:
        return "Positively skewed (right-skewed)"
    elif skewness < 0:
        return "Negatively skewed (left-skewed)"
    else:
        return "Approximately symmetric"


# Function 9: Identify Kurtosis
def identify_kurtosis(kurtosis):
    if kurtosis > 0:
        return "Leptokurtic (heavy-tailed)"
    elif kurtosis < 0:
        return "Platykurtic (light-tailed)"
    else:
        return "Mesokurtic (normal distribution)"


# Function 10: Main Function
def main():
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate normal distribution data
    test_results = generate_normal_distribution(mean=75, std_dev=10, size=1000)

    # Plot Histogram
    plot_histogram(test_results)

    # Plot PDF
    plot_pdf(test_results)

    # Plot CDF
    plot_cdf(test_results)

    # Calculate skewness and kurtosis
    skewness = calculate_skewness(test_results)
    kurt = calculate_kurtosis(test_results)

    # Print skewness and kurtosis
    print_statistics(skewness, kurt)

    # Identify skewness and kurtosis
    skewness_label = identify_skewness(skewness)
    kurtosis_label = identify_kurtosis(kurt)

    print("\nDistribution Characteristics:")
    print(f"Skewness: {skewness_label}")
    print(f"Kurtosis: {kurtosis_label}")


if __name__ == "__main__":
    main()
