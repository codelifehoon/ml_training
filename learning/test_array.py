import numpy as np
a = [1, 2, 3, ['a', 'b', 'c']]
print(a)
print(a[-1])
print(a[-1][0])


a = [1, 2, 3, 4, 5]
print(a[0:2])


array_data = np.arange(start=0,stop=100)
array_data = np.reshape(array_data,[-1,5])




# zero-axis(x) filter
print(array_data[2::2],np.size(array_data[2::2]))

# one-axis(y) filter
print(array_data[:,::2],np.size(array_data[:,::2]))

# [zero-axix,one-axis]
print(array_data[2::2,::2],np.size(array_data[2::2,::2]))


zero_num = np.zeros((5,))
zero_num[[0,2,4]]  = [0,2,4]
print(zero_num)


