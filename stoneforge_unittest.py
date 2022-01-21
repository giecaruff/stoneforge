
#%%

import unittest
import stoneforge
import numpy as np

class testTest(unittest.TestCase):

    """
    test several values for simandoux equation.
    """

    def test_simandoux(self):

        for i in range(0,10):
            with self.subTest(i=i):
                phi = np.random.uniform(0., 1.)
                vsh = np.random.uniform(0., 1.)
                a = np.random.uniform(.80, 1.)
                m = np.random.uniform(1.7, 2.)
                n = np.random.uniform(1.8, 2.)
                rt = np.random.uniform(1., 10000.)
                rw = np.random.uniform(0., 2.)
                rshale = np.random.uniform(2., 10.)
                result = stoneforge.petrophysics.water_saturation.simandoux(rw,rt,phi,a,m,n,vsh,rshale)
                #self.assertEqual(i % 2, 0) ### <- test response for visualization only
                self.assertTrue(result >= 0. and result <= 1.)


if __name__ == '__main__':
   log_file = 'log_file.txt'
   with open(log_file, "w") as f:
       runner = unittest.TextTestRunner(f)
       unittest.main(testRunner=runner)

#%%