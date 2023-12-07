# basic_matplotlib_functions.py

import matplotlib.pyplot as plt
import numpy as np

# 1. Line Plot
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 2. Scatter Plot
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, color='red', marker='o')
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 3. Bar Plot
categories = ['Category A', 'Category B', 'Category C']
values = [4, 7, 2]
plt.bar(categories, values, color='blue')
plt.title("Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

# 4. Histogram
data = np.random.randn(1000)
plt.hist(data, bins=20, color='green', alpha=0.7)
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# 5. Pie Chart
sizes = [30, 40, 20, 10]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen'])
plt.title("Pie Chart")
plt.show()

# 6. Box Plot
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(data, vert=True, patch_artist=True)
plt.title("Box Plot")
plt.xlabel("Data Sets")
plt.ylabel("Values")
plt.show()

# 7. Violin Plot
plt.violinplot(data, showmedians=True)
plt.title("Violin Plot")
plt.xlabel("Data Sets")
plt.ylabel("Values")
plt.show()

# 8. 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title("3D Scatter Plot")
plt.show()

# 9. Image Plot
image_data = np.random.random((10, 10))
plt.imshow(image_data, cmap='viridis', interpolation='nearest')
plt.title("Image Plot")
plt.colorbar()
plt.show()

# 10. Annotations
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.annotate('Peak', xy=(np.pi / 2, 1), xytext=(np.pi / 2, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# 11. Subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].plot(x, y, color='r')
axes[0, 1].scatter(x, y, color='g')
axes[1, 0].bar(categories, values, color='b')
axes[1, 1].hist(data[0], bins=20, color='y')
plt.show()

# 12. Twin Axes
fig, ax1 = plt.subplots()
ax1.plot(x, y, 'b-')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Primary Y-axis', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(x, np.cos(x), 'r-')
ax2.set_ylabel('Secondary Y-axis', color='r')
ax2.tick_params('y', colors='r')

plt.title("Twin Axes")
plt.show()

# 13. Log Scale
x = np.linspace(0.1, 10, 100)
y = np.log(x)
plt.plot(x, y)
plt.title("Logarithmic Scale")
plt.xlabel("X-axis")
plt.ylabel("Y-axis (log scale)")
plt.show()

# 14. Polar Plot
theta = np.linspace(0, 2*np.pi, 100)
r = 1.5 + np.sin(6*theta)
plt.polar(theta, r)
plt.title("Polar Plot")
plt.show()

# 15. Error Bars
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 3, 5])
error = np.array([0.1, 0.2, 0.1, 0.1, 0.3])
plt.errorbar(x, y, yerr=error, fmt='o-', color='purple')
plt.title("Error Bars")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 16. Color Maps
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar()
plt.title("Contour Plot with Color Map")
plt.show()
