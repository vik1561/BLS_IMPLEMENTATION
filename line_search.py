import numpy as np


class LineSearch:
    """
    Represents a 1D line search direction passing through base points RA and RB:
    Point(t) = RA + t * (RB - RA)
    """
    def __init__(self):
        self.direction = None
        self.RA = None
        self.RB = None

    def set_base_points(self, RA, RB):
        self.RA = np.array(RA, dtype=float)
        self.RB = np.array(RB, dtype=float)
        self.direction = self.RB - self.RA

    def point(self, t):
        return self.RA + t * self.direction
