'''
Standard Python distribution doesn't come bundled with NumPy module. A lightweight alternative is to install NumPy using popular Python package installer, pip.
'pip install numpy'
Otherwise, Python with distribution comes with 'numpy' package
'''

import numpy as np

print(np.version.version)
print(np.__version__)
'''
##Topic - 1
#########################START: NumPy -  Ndarray Object
#Creating an ndarray:   numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
#object: array interface method returns an array, or any (nested) sequence.
#dtype: optional / desired datatype of array
#copy: optional / default=True,  the object is copied
#order: optional / default=A . C (row major) or F (column major) or A (any) (default)
#subok: optional / default=False. By default, returned array forced to be a base class array. If true, sub-classes passed through
#ndmin: optional / default=0, Specifies minimum dimensions of resultant array

oneD_array = np.array([1, 2, ])
twoD_array = np.array([[1, 2], [3, 4]])
ndmin_param_array = np.array([1, 2, 3, 4, 5], ndmin=2)
complex_array = np.array([1, 2, 3], dtype=complex)


print(oneD_array) #Output: [1,2,3]
print(twoD_array) #Output: [[1 2] \n [3 4]]
print(ndmin_param_array) #Output: [[1, 2, 3, 4, 5]]
print(complex_array) #Output: [ 1.+0.j,  2.+0.j,  3.+

print(type(oneD_array)) #Output: <class 'numpy.ndarray'>
print(type(twoD_array)) #Output: <class 'numpy.ndarray'>

##########################END: NumPy - Ndarray Object
'''

##Topic - 2
##########################START: NumPy - Data Types

dt_int32 = np.dtype(np.int32)
print(dt_int32)  #Output: int32

#int8, int16, int32, int64 can be replaced by equivalent string 'i1', 'i2','i4', etc.
dt_i2 = np.dtype('i2')
print(dt_i2)   #Output: int16

#'<' is for little-endian, '>' is for big-endian. Default is little-endian
dt_be = np.dtype('>i4')
dt_le = np.dtype('<i4')
print(dt_le, dt_be) #Output: int32 >i4

#For structured data array, there could be several fields with different data types. Hence, we create, array of tuples.
dt_1structuredField = np.dtype([ ('age', np.int8) ])
print(dt_1structuredField)  #Output: [('age', 'i1')]

dt_multipleStructuredField = np.dtype([ ('name', 'S25'), ('age', np.int8),  ('salary', np.int32) ])
print(dt_multipleStructuredField)  #Output: [('name', 'S25'), ('age', 'i1'), ('salary', '<i4')]
employee_array = np.array([('Hary', 33, 10000 ), ('Sam', 29, 7000), ('Mike', 39, 11000)] , dtype = dt_multipleStructuredField )
print(employee_array)  #Output: [(b'Hary', 33, 10000) (b'Sam', 29,  7000) (b'Mike', 39, 11000)]
print(employee_array.ndim) #Output: 1

array_1Byte = np.array([1, 2, 3, 4, 5], dtype=np.int8)
print(array_1Byte.itemsize)   #Output: 1

array_4Byte = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print(array_1Byte.item)   #Output: 4

##########################END: NumPy - Data Types

##Topic - 3
##########################START: NumPy - Array Creation

#Array can be created either by providing data during creation or as empty, zeros or ones filled array.
#numpy.empty: numpy.empty(shape, dtype = float, order = 'C')
array_empty_float32 = np.empty([2, 3], dtype=np.float32)
print(array_empty_float32)   #Output: [[4.4659382e-42 1.9043008e-36 3.9129861e-34]  \n [2.5450012e-29 1.2735141e-40 0.0000000e+00]]

#numpy.zeros: numpy.zeros(shape, dtype = float, order = 'C')
array_zeros_2x3 = np.zeros([2, 3], dtype=np.float32)
print(array_zeros_2x3)   #Output: [[0. 0. 0.] \n [0. 0. 0.]]

#numpy.ones: numpy.ones(shape, dtype = float, order = 'C')
array_ones_2x3 = np.ones( (2, 3), dtype=np.float32)
print(array_ones_2x3)   #Output: [[1. 1. 1.] \n [1. 1. 1.]]
#When array is populated using command line and not explicity, then instead of comma separator, you see dot seprator.

##########################END: NumPy - Array Creation

##TOpic -4
##########################START: NumPy - Array Creation from existing data

#numpy.asarray : similar to numpy.array except for the fact that it has fewer parameters.
#Format: numpy.asarray(a, dtype = None, order = None)
x = [1, 2, 3]
a = np.asarray(x, dtype=np.float32)
print(a)   #Output:

#When array represented in matrix has it's rows and columns not consistent. .shape returns just fixed item.
x = [(1, 2, 3), (4, 5)]
a = np.asarray(x)
print(a)   #Output:
print(a.shape)  #Output: (2,)

#numpy.frombuffer : interprets a buffer as one-dimensional array. Any object that exposes the buffer interface is used as parameter to return an ndarray.
#Format: numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
#A string isn't a buffer in Python3.x. The default string type is unicode. The 'b' is used to create and display bytestrings.
s = b'Hello World'
a = np.frombuffer(s, dtype='S1')
print(a)   #Outut: [b'H' b'e' b'l' b'l' b'o' b' ' b'W' b'o' b'r' b'l' b'd']

#numpy.fromiter : builds an ndarray object from any iterable object. A new one-dimensional array is returned by this function.
#Format: numpy.fromiter(iterable, dtype, count = -1)

list = range(5)
print(list)  #Output: range(0, 5) or [0,  1,  2,  3,  4]
it = iter(list)

# use iterator to create ndarray
x = np.fromiter(it, dtype=np.float32)
print(x)   #Ouput: [0. 1. 2. 3. 4.]

##########################END: NumPy - Array Creation from existing data


##TOpic -5
##########################START: NumPy - Array Creation from Numerical Ranges

##numpy.arrange: returns an ndarray object containing evenly spaced values within a given range.
##Format:  numpy.arange(start, stop, step, dtype)
#start: optional / default 0
#end: required
#step: optional / default 1

x = np.arange(5)
print(x)  #Output: [0  1  2  3  4]
x = np.arange(5, dtype = float)
print(x)  #Output: [0.  1.  2.  3.  4.]
x = np.arange(10,20,2)
print(x)  #Output: [10  12  14  16  18]


##numpy.linspace : function is similar to arange() function. In this function, instead of step size, the number of evenly spaced values between the interval is specified.
##Format: numpy.linspace(start, stop, num, endpoint, retstep, dtype)
x = np.linspace(10,20,5)
print(x)  #Output: [10.   12.5   15.   17.5  20.]
x = np.linspace(10,20, 5, endpoint = False)
print(x)   #Output: [10.   12.   14.   16.   18.]
x = np.linspace(1,2,5, retstep = True)
print(x)   #Output: (array([ 1.  ,  1.25,  1.5 ,  1.75,  2.  ]), 0.25)

##numpy.logspace:  function returns an ndarray object that contains the numbers that are evenly spaced on a log scale
##Format:  numpy.logspace(start, stop, num, endpoint, base, dtype)
# default base is 10
a = np.logspace(1.0, 2.0, num = 10)
print(a)  #Output: [ 10.          12.91549665  16.68100537  21.5443469   27.82559402   35.93813664  46.41588834  59.94842503  77.42636827 100.]

##########################END: NumPy - Array Creation from Numerical Ranges


##Topic - 6
##########################START: NumPy - Indexing and Slicing
#slice(start, stop, step)
a = np.arange(10)
s = slice(2, 7, 2)
print(a[s])   #Output: [2  4  6]

a = np.arange(10)
b = a[2:7:2]
print(b)   #Output: [2  4  6]

a = np.arange(10)
print(a[2:])  #Output: [2  3  4  5  6  7  8  9]

a = np.arange(10)
print(a[2:5])   #Output: [2  3  4]

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a[1:])  #Output: [[3 4 5] \n  [4 5 6]]

##########################END: NumPy - Indexing and Slicing

##Topic - 7
##########################START: NumPy - Advanced Indexing
#There are two types of advanced indexing − Integer and Boolean.

#Integer Indexing: one element of specified column from each row of ndarray object is selected. Hence, the row index contains all row numbers, and the column index specifies the element to be selected.
x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0,1,2], [0,1,0]]
print(y)  #Output: [1  4  5] because The selection includes elements at (0,0), (1,1) and (2,0) from the first array.

#Boolean Indexing: This type of advanced indexing is used when the resultant object is meant to be the result of Boolean operations, such as comparison operators.
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])
print (x[x > 5])  #Output: [ 6  7  8  9 10 11]

##########################END: NumPy - Advanced Indexing

##Topic - 8
##########################START: NumPy - Broadcasting


##########################END: NumPy - Broadcasting


