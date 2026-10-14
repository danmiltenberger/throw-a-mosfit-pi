from dataclasses import dataclass




@dataclass
class Point:
    """A point in 2D space, in inches or pixels."""

    x: float
    y: float

  

@dataclass
class Rectangle:
    """Rectangle geometry following OpenCV's x, y, width, height convention."""

    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 100



@dataclass 
class OurRobot:
    position_in : Point
    azimuth_rad : float


@dataclass
class Detection:
    """structure that describes the object in the frame"""

    # from the camera
    name = str
    width_px = int
    center_px = Point

    # updated with inch range and angle
    range = float
    azimuth_rad = float

    # update to a global position
    pos_global_in = Point
    pos_global_px = Point