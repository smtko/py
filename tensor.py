# basic_tensorflow_functions.py
import numpy as np
import tensorflow as tf

# Function 1: Create a Constant Tensor
tensor_1 = tf.constant(5.0)

# Function 2: Create a Variable Tensor
variable_1 = tf.Variable(3.0)

# Function 3: Perform Addition
sum_tensor = tensor_1 + variable_1

# Function 4: Perform Matrix Multiplication
matrix_a = tf.constant([[1, 2], [3, 4]])
matrix_b = tf.constant([[5, 6], [7, 8]])
product_matrix = tf.matmul(matrix_a, matrix_b)

# Function 5: Define a Placeholder
placeholder_a = tf.placeholder(tf.float32)
placeholder_b = tf.placeholder(tf.float32)
sum_placeholders = placeholder_a + placeholder_b

# Function 6: Create a Session
with tf.Session() as session:
    # Function 7: Run a Tensor in the Session
    result_tensor = session.run(sum_tensor)
    print("Result of Tensor Addition:", result_tensor)

    # Function 8: Initialize Variables
    session.run(tf.global_variables_initializer())
    result_variable = session.run(variable_1)
    print("Value of Variable:", result_variable)

    # Function 9: Run Matrix Multiplication
    result_matrix = session.run(product_matrix)
    print("Result of Matrix Multiplication:")
    print(result_matrix)

    # Function 10: Feed Values to Placeholders
    result_placeholders = session.run(sum_placeholders, feed_dict={placeholder_a: 2.0, placeholder_b: 3.0})
    print("Result of Placeholder Addition:", result_placeholders)

# Function 11: Create a TensorFlow Constant Tensor with Numpy
numpy_array = tf.constant(np.array([1.0, 2.0, 3.0]))

# Function 12: Define a TensorFlow Operation
operation_1 = tf.add(2, 3)

# Function 13: Perform Element-wise Multiplication
vector_a = tf.constant([1, 2, 3])
vector_b = tf.constant([4, 5, 6])
elementwise_product = tf.multiply(vector_a, vector_b)

# Function 14: Reduce Operation (Sum)
sum_elements = tf.reduce_sum(vector_a)

# Function 15: Placeholder with Shape
placeholder_shape = tf.placeholder(tf.float32, shape=(None, 3))

# Run the TensorFlow Session
with tf.Session() as session:
    result_numpy = session.run(numpy_array)
    print("TensorFlow Constant Tensor with Numpy:")
    print(result_numpy)

    result_operation = session.run(operation_1)
    print("Result of TensorFlow Operation:")
    print(result_operation)

    result_elementwise_product = session.run(elementwise_product)
    print("Result of Element-wise Multiplication:")
    print(result_elementwise_product)

    result_sum_elements = session.run(sum_elements)
    print("Result of Reduce Operation (Sum):", result_sum_elements)

    # Placeholder with Shape
    result_placeholder_shape = session.run(placeholder_shape, feed_dict={placeholder_shape: np.array([[1, 2, 3], [4, 5, 6]])})
    print("Result of Placeholder with Shape:")
    print(result_placeholder_shape)
