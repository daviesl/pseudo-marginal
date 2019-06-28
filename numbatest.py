import numpy as np
import numba
from numba import cuda
from numba import guvectorize
from timeit import default_timer as timer
from scipy.special import logsumexp
import math
import sys
#m=-sys.float_info.max

@cuda.reduce
def sum_reduce(a, b):
    return a + b

#@cuda.reduce
#def max_reduce(a, b):
#    return max(a,b)

@cuda.jit(device=True)
def my_max(result, values):
    """Find the maximum value in values and store in result[0]"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    cuda.atomic.max(result, 0, values[i])

@cuda.jit(device=True)
def my_sum(result, values):
    """Find the maximum value in values and store in result[0]"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    cuda.atomic.add(result, 0, values[i])

#@numba.vectorize(['float64(float64)'], target='cuda')
#def exp_gpu(x):
#    return math.exp(x)


#@cuda.jit(device=True)
#def max_gpu(x):
#    m=x[0]
#    for i in range(0,x.shape[0]):
#        m=max(m,x[i])
#    return m
        

@cuda.jit(device=True)
def logsumexp_gpu(ns,out,m):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    #m = max_gpu(ns)
    #tmp = cuda.local.array(ns.shape)
    for i in range(start, ns.shape[0], stride):
        ns[i] = math.exp(ns[i] - m)
    my_sum(out,ns)
    #m = np.zeros(1,dtype=np.float64)
    #my_max(m,ns)
    #return math.log(my_sum(np.exp(ns-np.max(ns)))) + np.max(ns)

@cuda.reduce
def constsumexp_reduce(x,m):
    return math.exp(x + m)

@cuda.jit
def logsumexp(a, b):
    return a + b


@cuda.jit
def run_logsumexp_gpu(ns,out,m):
    logsumexp_gpu(ns,out,m)

A = np.log((np.arange(123456789, dtype=np.float64)) + 1)
start = timer() 
#got = sum_reduce(A)   # cuda sum reduction
got = np.zeros((1,1),dtype=np.float64)
m = np.zeros((1,1),dtype=np.float64)
m[:] = np.max(A)
threadsperblock=32
blockspergrid=(A.size + (threadsperblock-1))
run_logsumexp_gpu[blockspergrid,threadsperblock](A,got,m)   # cuda sum reduction
print(got)
print("CUDA elapsed time: {}".format( timer() - start ))
start = timer() 
expect = logsumexp(A)      # np sum reduction
print(expect)
print("Numpy elapsed time: {}".format( timer() - start ))
assert expect == got
