from numpy import genfromtxt
import numpy as np



my_data = genfromtxt('my_data_vol.csv', delimiter=',')
my_data = np.array(my_data)
my_data = my_data[1:]
y_test = my_data[:, 5]
x_test = my_data[:, 0:5]

print(y_test)