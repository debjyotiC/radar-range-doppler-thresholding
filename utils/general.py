import numpy as np
import scipy.signal as signal
import os

def moving_average_2d_1(a, n=3):
    ret = np.cumsum(a, dtype=float, axis=1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n

def moving_average_2d_2(a, n=3):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:, :] = ret[n:, :] - ret[:-n, :]
    return ret[n - 1:, :] / n

def convolution_1d(a, n=10, mode='same'):
    kernel = np.ones((n,))/n
    return np.convolve(a, kernel, mode = mode)

def convolution_2d(a, n=10, mode='same', boundary='symm'):
    kernel = np.ones((n, n))/(n*n)
    return signal.convolve2d(a, kernel, mode = mode, boundary=boundary)

# import numpy as np

# # Define the input 2D array (image) and the kernel (filter)
# image = np.random.rand(100,100)
# print(image[:10,:10])

# kernel = np.array([1, 1, 1])

# # Perform the 2D convolution
# result =convolution_2d(image)

# print(result.shape)
# print(result[:10,:10])
# # print(result1)
    