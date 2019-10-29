import numpy as np
import numba

@numba.jit #("float64[:](float64[:][:],float64[:][:],float64[:][:])")
def logmvnorm_vectorised(X,mu,cov):
    return -0.5 * math.log(2. * math.pi) - 0.5 * math.log(np.linalg.det(cov)) - 0.5 * np.sum(np.dot(X - mu,np.linalg.inv(cov)) * (X-mu), axis=1)


@numba.jit("f8(f8[:])")
def logsumexp(ns):
    m = np.max(ns)
    ds = ns - m
    sumOfExp = np.exp(ds).sum()
    return m + np.log(sumOfExp)

@numba.jit
def logsumexp_pair(ns1,ns2):
    m = np.maximum(ns1,ns2)
    ds1 = ns1 - m
    ds2 = ns2 - m
    sumOfExp = np.exp(ds1) + np.exp(ds2)
    return m + np.log(sumOfExp)

#@numba.jit
def logsumexp_mat(ns,axis=0):
    m = np.max(ns,axis=axis)
    ds = ns - m[:,np.newaxis]
    sumOfExp = np.exp(ds).sum(axis=axis)
    return m + np.log(sumOfExp)

@numba.jit("f8[:,:](f8[:,:],f8[:,:])")
def doubledot(outer,inner):
    """
    returns F C F' where F is a state to observation matrix
    """
    return np.dot(np.dot(outer,inner),outer.T)

@numba.jit(nopython=True)
def mdla_dottrail2x2(A,B):
    """
    The idea is to run a dot product on the last k dimensions
    of two matrices. This way the first 0 to n-k dimensions
    of the multidimensional matrices can be preserved.
    Somehow require that dim(A) >= 2 and dim(B) >= 2
    and dim(A)==dim(B)
    and A.shape[:-2] == B.shape[:-2]
    and A.shape[-1] == B.shape[-2]
    """
    m = A.shape[-2]
    n = A.shape[-1]
    p = B.shape[-1]
    Cshape = A.shape[:-2] + (m,p)

    C = np.zeros(Cshape)

    for i in range(0,m):
        for j in range(0,p):
            for k in range(0,n):
                C[...,i,j] += A[...,i,k]*B[...,k,j] 
    return C

@numba.jit(nopython=True)
def mdla_dottrail2x1(A,B):
    """
    The idea is to run a dot product on the last k dimensions
    of two matrices. This way the first 0 to n-k dimensions
    of the multidimensional matrices can be preserved.
    Somehow require that dim(A) >= 2 and dim(B) >= 1
    and dim(A)==dim(B)
    and A.shape[:-2] == B.shape[:-2]
    and A.shape[-1] == B.shape[-2]
    """
    m = A.shape[-2]
    n = A.shape[-1]
    Cshape = A.shape[:-2] + (m,)

    C = np.zeros(Cshape)

    for i in range(0,m):
        for k in range(0,n):
            C[...,i] += A[...,i,k]*B[...,k]
    return C

@numba.jit
def xTPx(x,P):
    """
    A helper method to compute the exponent term of the Gaussian
    for multivariate (x dim > 1) and a system of N particles in the
    last(first?) dimension
    """
    return np.sum(np.dot(x,P)*x,axis=1)


@numba.jit('f8[:,:,:](f8[:,:],UniTuple(i8,3))',nopython=True)
def mdla_broadcast_to(a,shape):
    return np.ones(shape) * a
    #l=len(a.shape)
    #return numba_flattile(a.ravel(),shape[:-l]).reshape(nccat(shape[:-l],nshp(a)))
    #return numba_flattile(a.ravel(),shape[:-l]).reshape(shape[:-l]+a.shape)
    #return np.tile(a.ravel(),shape[:-l]).reshape(shape[:-l]+a.shape)

#@numba.jit
#def nshp(a):
#    return np.array(a.shape)

#@numba.jit
#def nccat(a,b):
#    return np.concatenate((a,b))

#@numba.jit(nopython=True)
@numba.jit('f8[:,:,:](f8[:,:],f8[:,:,:])',nopython=True)
def mdla_dottrail2x2_broadcast(A,B):
    # broadcast A
    shape = B.shape[:-2]+A.shape[-2:]
    Aa = mdla_broadcast_to(A,shape)
    return mdla_dottrail2x2(Aa,B)

@numba.jit('f8[:,:](f8[:,:],f8[:,:])',nopython=True)
def mdla_dottrail2x1_broadcast(A,B):
    # broadcast A
    Ashape = B.shape[:-1]+A.shape[-2:]
    Aa = mdla_broadcast_to(A,Ashape)
    return mdla_dottrail2x1(Aa,B)

#@numba.jit('f8[:,:,:](f8[:,:,:],f8[:,:,:])',nopython=True)
#def mdla_dottrail2x2_broadcast(A,B):
#    return mdla_dottrail2x2(A,B)

#def mdla_dottrail2x2_broadcast(A,B):
#    if A.ndim > B.ndim:
#        # broadcast B
#        shape =A.shape[:-2]+B.shape[-2:]
#        B_ = mdla_broadcast_to(B,shape)
#        #B = np.broadcast_to(B,A.shape[:-2]+B.shape[-2:])
#        return mdla_dottrail2x2(A,B_)
#    elif B.ndim > A.ndim:
#        # broadcast A
#        shape = B.shape[:-2]+A.shape[-2:]
#        A_ = mdla_broadcast_to(A,shape)
#        #A = np.broadcast_to(A,B.shape[:-2]+A.shape[-2:])
#        return mdla_dottrail2x2(A_,B)
#    else:
#        return mdla_dottrail2x2(A,B)
    

#TODO numba tile
@numba.jit
def numba_flattile(a,shape):
    #s1=np.array(shape).sum()
    #s2=np.array(a.ravel().shape).sum()
    #return np.ones(nccat(shp,nshp(a)).tolist())*a
    return np.ones(shape+a.shape)*a
    #return np.ones(s1*s2).reshape((s1,)+a.shape)*a

@numba.jit
def doublemdla_dottrail2x2_broadcast(outer,inner):
    """
    returns F C F' where F is a state to observation matrix
    """
    return mdla_dottrail2x2_broadcast(mdla_dottrail2x2_broadcast(outer,inner),outer.T)

@numba.jit(nopython=True)
def mdla_invtrail2d(A):
    A_ = A.copy().reshape((-1,)+A.shape[-2:])
    Ainv = np.zeros_like(A_)
    for k in range(A_.shape[0]):
        Ainv[k,...] = np.linalg.inv(A_[k,...])
    return Ainv.reshape(A.shape)

#mdla3d_invtrail2d = numba.jit('f8[:,:,:](f8[:,:,:])',nopython=True)(mdla_invtrail2d_nojit)
#mdla4d_invtrail2d = numba.jit('f8[:,:,:,:](f8[:,:,:,:])',nopython=True)(mdla_invtrail2d_nojit)

# The below fails due to type error on recursion
#@numba.jit(nopython=True)
#def mdla_invtrail2d(A):
#    if len(A.shape) > 2:
#        Ainv = np.zeros_like(A)
#        for k in range(A.shape[0]):
#            Ainv[k,...] = mdla_invtrail2d(A[k,...])
#        return Ainv
#    else:
#        return np.linalg.inv(A)

@numba.jit
def doublemdla_dottrail2x2(outer,inner):
    """
    returns F C F' where F is a state to observation matrix
    """
    return mdla_dottrail2x2(mdla_dottrail2x2(outer,inner),outer.T)

if __name__ == '__main__': 
    a,b,c = np.mgrid[1:5:1, 1:3:1, 1:4:1] 
    d,e,f = np.mgrid[1:5:1, 1:4:1, 1:3:1] 
    A = a*b*c
    B = d*e*f
    F = np.array([[1,0,0],[0,0,1]])
    print("F = {}".format(F))
    print("A = {}\nB = {}".format(A,B))
    #tileF = numba_flattile(F,np.array((4,)))
    tileF = numba_flattile(F,np.array([4]))
    print("tiled F = {}".format(tileF))
    bF = mdla_broadcast_to(F,(5,2,3))
    print("broadcast F = {}".format(bF))
    C = mdla_dottrail2x2_broadcast(F,B)
    print("dot(F,B) = {}".format(C))
    C = mdla_dottrail2x2_broadcast(A,B)
    print("dot(A,B) = {}".format(C))
    C = np.dot(A,B)
    print("numpy.dot(A,B) = {}".format(C))
    C = mdla_dottrail2x2(B,A)
    print("dot(B,A) = {}".format(C))
    C = np.dot(B,A)
    print("numpy.dot(B,A) = {}".format(C))
    # now try with a single dot mat
    C = mdla_dottrail2x2_broadcast(A[0,...],B)
    print("dot(A[0,...],B) = {}".format(C))
    
