# basic_scipy_functions.py

import numpy as np
from scipy import stats, optimize, interpolate, integrate, signal, linalg, spatial, special

# 1. Descriptive Statistics - Mean
data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)
print("Mean:", mean_value)

# 2. Hypothesis Testing - t-test
t_stat, p_value = stats.ttest_1samp(data, 3)
print("t-statistic:", t_stat)
print("p-value:", p_value)

# 3. Optimization - Minimization
result = optimize.minimize(lambda x: (x - 3)**2, x0=0)
print("Minimization Result:", result.x)

# 4. Interpolation - 1D
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 1, 4])
f = interpolate.interp1d(x, y, kind='linear')
print("Interpolated Value at x=2.5:", f(2.5))

# 5. Integration
result = integrate.quad(lambda x: x**2, 0, 1)
print("Integral Result:", result[0])

# 6. Signal Processing - Convolution
signal1 = np.array([1, 2, 3])
signal2 = np.array([0, 1, 0.5])
conv_result = signal.convolve(signal1, signal2, mode='full')
print("Convolution Result:", conv_result)

# 7. Linear Algebra - Matrix Inversion
matrix = np.array([[1, 2], [3, 4]])
inverse_matrix = linalg.inv(matrix)
print("Inverse Matrix:\n", inverse_matrix)

# 8. Spatial - Distance between Points
point1 = np.array([1, 2])
point2 = np.array([4, 6])
distance = spatial.distance.euclidean(point1, point2)
print("Distance between Points:", distance)

# 9. Special Functions - Bessel Function
bessel_value = special.jn(2, 1.5)
print("Bessel Function J_2(1.5):", bessel_value)

# 10. Probability Distribution - Normal Distribution
rv = stats.norm(loc=0, scale=1)
random_samples = rv.rvs(size=5)
print("Random Samples from Normal Distribution:", random_samples)

# 11. Cumulative Distribution Function (CDF)
cdf_value = rv.cdf(0)
print("CDF at x=0:", cdf_value)

# 12. Percentile Calculation
percentile_value = np.percentile(data, 75)
print("75th Percentile:", percentile_value)

# 13. Pearson Correlation Coefficient
correlation_coefficient, _ = stats.pearsonr(x, y)
print("Pearson Correlation Coefficient:", correlation_coefficient)

# 14. Fourier Transform
time = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 5 * time)
fft_result = np.fft.fft(signal)
print("Fourier Transform Result:", fft_result)

# 15. Ordinary Differential Equation (ODE) Solving
def ode_func(y, t):
    return -2 * y

ode_solution = integrate.odeint(ode_func, y0=1, t=np.linspace(0, 1, 100))
print("ODE Solution:", ode_solution)

# 16. Non-Linear Equation Solving
root_result = optimize.root(lambda x: x**2 - 4, x0=0)
print("Root of x^2 - 4:", root_result.x)
