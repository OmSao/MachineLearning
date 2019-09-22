import numpy as np

oneD_array = np.array([1, 2, ])
twoD_array = np.array([[1, 2], [3, 4]])
ndmin_param_array = np.array([1, 2, 3, 4, 5], ndmin=2)
complex_array = np.array([1, 2, 3], dtype=complex)


print(oneD_array) #Output: [1,2,3]
print(twoD_array) #Output: [[1 2] \n [3 4]]
print(ndmin_param_array) #[[1, 2, 3, 4, 5]]
print(complex_array) #[ 1.+0.j,  2.+0.j,  3.+

print(type(oneD_array)) #Output: <class 'numpy.ndarray'>
print(type(twoD_array)) #Output: <class 'numpy.ndarray'>

