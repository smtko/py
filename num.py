import numpy as np

# 1. Create a 1D Array
array_1d = np.array([1, 2, 3, 4, 5])
print("1. 1D Array:", array_1d)

# 2. Create a 2D Array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2. 2D Array:\n", array_2d)

# 3. Array Shape
shape_2d = array_2d.shape
print("3. Array Shape:", shape_2d)

# 4. Array Dimensions
dimensions_2d = array_2d.ndim
print("4. Array Dimensions:", dimensions_2d)

# 5. Array Size
size_2d = array_2d.size
print("5. Array Size:", size_2d)

# 6. Array Data Type
dtype_2d = array_2d.dtype
print("6. Array Data Type:", dtype_2d)

# 7. Reshape Array
reshaped_array = array_1d.reshape((5, 1))
print("7. Reshaped Array:\n", reshaped_array)

# 8. Slicing
sliced_array = array_1d[1:4]
print("8. Sliced Array:", sliced_array)

# 9. Element-wise Operations
addition_result = array_1d + 10
print("9. Element-wise Addition Result:", addition_result)

# 10. Dot Product
dot_product_result = np.dot(array_1d, array_1d)
print("10. Dot Product Result:", dot_product_result)

# 11. Transpose
transposed_array = array_2d.T
print("11. Transposed Array:\n", transposed_array)

# 12. Sum along Axis
sum_along_axis_0 = np.sum(array_2d, axis=0)
print("12. Sum along Axis 0:", sum_along_axis_0)

# 13. Statistical Functions
mean_value = np.mean(array_1d)
std_deviation = np.std(array_1d)
print("13. Mean Value:", mean_value)

# 14. Array Concatenation
concatenated_array = np.concatenate((array_1d, array_1d))
print("14. Concatenated Array:", concatenated_array)

# 15. Index of Maximum Value
index_max_value = np.argmax(array_1d)
print("15. Index of Maximum Value:", index_max_value)

# 16. Generating Random Numbers
random_array = np.random.rand(3, 3)
print("16. Random Array:\n", random_array)

