import unittest
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

    def test_mdla_dottrail2x1_broadcast(self):
        Ar = self.A[:,0,:]
        C = mdla_dottrail2x1_broadcast(self.F,Ar)
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
