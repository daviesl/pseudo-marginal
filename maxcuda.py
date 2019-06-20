from numba import cuda
import numpy as np
from timeit import default_timer as timer

@cuda.jit
def max_example(result, values):
    """Find the maximum value in values and store in result[0]"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    cuda.atomic.max(result, 0, values[i])


arr = np.random.rand(2**26)
result = np.zeros(1, dtype=np.float64)

start = timer()
max_example[256,64](result, arr)
print("Cuda time = {}".format(timer()-start))
print(result[0]) # Found using cuda.atomic.max

start = timer()
print(max(arr))  # Print max(arr) for comparision (should be equal!)
print("CPU time = {}".format(timer()-start))
