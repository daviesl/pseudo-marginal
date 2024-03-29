import unittest
import sys
from mdlinalg import *

class TestMDLinAlg(unittest.TestCase):

    def setUp(self):
        a,b,c = np.mgrid[1:5:1, 1:3:1, 1:4:1] 
        d,e,f = np.mgrid[1:5:1, 1:4:1, 1:3:1] 
        self.A = 1.0 * a*b*c
        self.B = 1.0 * d*e*f
        self.F = np.array([[1.,0.,0.],[0.,0.,1.]])

    def test_flattile(self):
        tileF = numba_flattile(self.F,(4,))
        self.assertSequenceEqual(tileF.ravel().tolist(), np.tile(self.F.ravel(),(1,4)).ravel().tolist())

    def test_broadcast_to(self):
        bF = mdla_broadcast_to(self.F,(5,2,3))
        self.assertSequenceEqual(bF.tolist(),np.broadcast_to(self.F,(5,2,3)).tolist())

    def test_mdla_dottrail1x2(self):
        Ar = self.A[:,0,:]
        bF = mdla_broadcast_to(self.F,self.A.shape)
        C = mdla_dottrail1x2(Ar,bF)
        #print("Ar={}\nbF={}\n1x2={}".format(Ar,bF,C))
        self.assertSequenceEqual(C.tolist(),self.A[:,(0,),(0,2)].tolist())

    def test_mdla_dottrail2x1_broadcast(self):
        Ar = self.A[:,0,:]
        C = mdla_dottrail2x1_broadcast(self.F,Ar)
        self.assertSequenceEqual(C.tolist(),self.A[:,(0,),(0,2)].tolist())

    def test_mdla_dottrail2x1_broadcast_full(self):
        Ar = self.A[:,0,:]
        bF = mdla_broadcast_to(self.F,self.A.shape)
        #print("Ar={}\nbF={}".format(Ar,bF))
        C = mdla_dottrail2x1_broadcast(bF,Ar)
        self.assertSequenceEqual(C.tolist(),self.A[:,(0,),(0,2)].tolist())

    def test_mdla_dottrail2x2_broadcast(self):
        C = mdla_dottrail2x2_broadcast(self.F,self.B)
        self.assertSequenceEqual(C.tolist(),self.B[:,(0,2),:].tolist())

    def test_mdla_dottrail2x2(self):
        C = mdla_dottrail2x2(self.A,self.B)
        trueC = np.ones((4,2,2))
        for i in range(4):
            trueC[i,...] = np.dot(self.A[i,...],self.B[i,...])
        self.assertSequenceEqual(C.tolist(),trueC.tolist())

    def test_mdla_invtrail2d(self):
        #C = mdla_dottrail2x2(self.A,self.B)
        C = np.broadcast_to(np.eye(3),(4,3,3))*np.arange(1,4)
        #print(C)
        #print("C.ndim = {}".format(C.ndim))
        #print("New shape = {}".format((-1,)+C.shape[-2:]))
        Ctest = mdla_invtrail2d(C)
        Cinv = np.ones_like(C)
        for i in range(4):
            Cinv[i,...] = np.linalg.inv(C[i,...])
        self.assertSequenceEqual(Ctest.tolist(),Cinv.tolist())

    def test_mdla_dettrail2d(self):
        #C = mdla_dottrail2x2(self.A,self.B)
        C = np.broadcast_to(np.eye(3),(4,3,3))*np.arange(1,4)
        #print(C)
        #print("C.ndim = {}".format(C.ndim))
        #print("New shape = {}".format((-1,)+C.shape[-2:]))
        #print("Det shape = {}".format(C.shape[:-2]))
        Ctest = mdla_dettrail2d(C)
        Cdet = np.ones(C.shape[:-2])
        #print("Cdet.shape = {}".format(Cdet.shape))
        for i in range(4):
            Cdet[i,...] = np.linalg.det(C[i,...])
        self.assertSequenceEqual(Ctest.tolist(),Cdet.tolist())

    def test_logmvnorm_vectorised(self):
        covar = np.broadcast_to(np.eye(3)*3.9+np.ones((3,3))*0.1,(4,3,3))
        #print("Cov={}".format(covar))
        X = np.outer(np.arange(0.8,1.2,0.1),np.arange(1,4))
        mu = np.zeros(3)
        ptest = logmvnorm_vectorised(X,mu,covar)
        #check
        from scipy.stats import multivariate_normal
        pcheck = np.ones(4)*0.1
        for i in range(4):
            #print("X={},\ncov={}".format(X[i,...],covar[i,...]))
            pcheck[i] = np.log(multivariate_normal.pdf(X[i,...],mean=mu,cov=covar[i,...]))
        #print("mdla p={}\nscipy p={}".format(ptest,pcheck))
        np.testing.assert_almost_equal(pcheck, ptest, decimal=13, err_msg='', verbose=True)
        #self.assertAlmostEqual(ptest.tolist(),pcheck.tolist())
        
    def test_xTPx_indivcov(self):
        covar = np.broadcast_to(np.eye(3)*3.9+np.ones((3,3))*0.1,(4,3,3))
        X = np.outer(np.arange(0.8,1.2,0.1),np.arange(1,4))
        r = xTPx_indivcov(X,covar)
        #print("xTPx={}".format(r))
        np.testing.assert_almost_equal([37.24800000000001, 47.142, 58.2, 70.42200000000001], r, decimal=13, err_msg='', verbose=True)
        #self.assertSequenceEqual(r.tolist(),[37.24800000000001, 47.142, 58.2, 70.42200000000001])

    def test_spsd_sqrtm(self):
        r = spsd_sqrtm(np.eye(4)*16+np.ones((4,4)))
        tr= np.array([[4.11803399, 0.11803399, 0.11803399, 0.11803399],
                   [0.11803399, 4.11803399, 0.11803399, 0.11803399],
                   [0.11803399, 0.11803399, 4.11803399, 0.11803399],
                   [0.11803399, 0.11803399, 0.11803399, 4.11803399]])
        np.testing.assert_almost_equal(tr,r,decimal=7,err_msg='Failed spsd_sqrtm()', verbose=True)

    def test_spsd_sqrtm_2(self):
        A = np.random.rand(40,40)
        B = np.dot(A,A.transpose())
        r = spsd_sqrtm(B)
        rr = np.dot(r,r.T)
        #np.set_printoptions(threshold=sys.maxsize)
        #print("r = {}".format(r))
        np.testing.assert_almost_equal(rr,B,decimal=7,err_msg='Failed spsd_sqrtm()', verbose=True)

    def test_spsd_sqrtm_3(self):
        r = spsd_sqrtm(np.eye(4)*0.0001)
        tr = np.eye(4)*0.01
        np.testing.assert_almost_equal(tr,r,decimal=7,err_msg='Failed spsd_sqrtm()', verbose=True)

    def test_spsd_sqrtm_vectorised(self):
        trk= np.array([[4.11803399, 0.11803399, 0.11803399, 0.11803399],
                   [0.11803399, 4.11803399, 0.11803399, 0.11803399],
                   [0.11803399, 0.11803399, 4.11803399, 0.11803399],
                   [0.11803399, 0.11803399, 0.11803399, 4.11803399]])
        tr = np.zeros((10,4,4))
        for i in range(10):
            tr[i,...]=trk
        r=spsd_sqrtm(np.broadcast_to(np.eye(4),(10,4,4))*16+np.ones((4,4)))
        np.testing.assert_almost_equal(tr,r,decimal=7,err_msg='Failed spsd_sqrtm()', verbose=True)
        
        

    #def test_someasserts(self):
        #self.assertTrue('FOO'.isupper())
        #self.assertFalse('Foo'.isupper())
        #with self.assertRaises(TypeError):
        #    s.split(2)

if __name__ == '__main__':
    unittest.main()

"""
Hilariously a lot of things in Numba fail.
For instance
@numba.jit 
def f(a): 
    return np.ones(a) 
f(np.array((3,2,3)))  
"""
