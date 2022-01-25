import unittest
from water_saturation import water_saturation


class SWTest(unittest.TestCase):
    def test_archie(self):
        self.assertAlmostEqual(water_saturation(rw=0.9, rt=20, phi=0.33,
                                                a=0.62, m=2.15, n=2.0,
                                                method="archie"), 0.55,
                                                places=2)


    def test_simandoux(self):
        self.assertAlmostEqual(water_saturation(rw=0.015, rt=1.0, phi=0.11,
                                                a=0.62, m=2.15, n=2.0,
                                                method="simandoux", vsh=0.33,
                                                rsh=4.0), 0.82, places=2)


if __name__ == '__main__':
    unittest.main()
