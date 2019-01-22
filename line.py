import math
from lane import Lane

class Line:

    """ Represent line and some useful methods

        Line: y = ax + b
        Represented by a pair of coordinates (x1, y1), (x2, y2)
    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.a = self.compute_slope()
        self.b = self.compute_intercept()

    def __repr__(self):
        return "Line: x1={x1}, " \
               "y1={y1}, " \
               "x2={x2}, " \
               "y2={y2}, " \
               "length={length}, " \
               "slope={slope}, " \
               "intercept={intercept}".format(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            length=self.length(),
            slope=self.a,
            intercept=self.b
        )

    def length(self):
        return math.sqrt((self.y2 - self.y1) ** 2 + (self.x2 - self.x1) ** 2)

    def get_coords(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def get_x(self, y):
        return ((self.b - y) // self.a)

    def get_y(self, x):
        return int(self.a * x + self.b)

    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def compute_intercept(self):
        return self.y1 - self.a * self.x1


    @property
    def candidate(self):
        """
        A simple domain logic to check whether this hough line can be a candidate
        for being a segment of a lane line.
        1. The line cannot be horizontal and should have a reasonable slope.
        2. The difference between lane line's slope and this hough line's cannot be too high.
        3. The hough line should not be far from the lane line it belongs to.
        4. The hough line should be below the vanishing point.
        """
        if abs(self.a) < Lane.MOSTLY_HORIZONTAL_SLOPE: return False
        # lane_line = getattr(Lane, self.lane_line)
        # if lane_line:
        #     if abs(self.a - lane_line.coeffs[0]) > Lane.MAX_SLOPE_DIFFERENCE: return False
        #     if self.distance_to_lane_line > Lane.MAX_DISTANCE_FROM_LINE: return False
        #     if self.y2 < Lane.left_line.vanishing_point[1]: return False
        return True

if __name__ == "__main__":
    l = Line(x1=0, y1=5, x2=1000, y2=150)
    print(l.candidate)
