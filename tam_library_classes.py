from dataclasses import dataclass, field
from typing import Optional
from tam_library_colors import YELLOW, CYAN
import numpy as np

# measure ccw from positive x (3oclock)
RADIANS_NORTH = np.pi/2
RADIANS_SOUTH = -np.pi/2
RADIANS_SOUTH_EAST = -np.pi/4
RADIANS_NORTH_EAST = np.pi/4



@dataclass
class Point:
    """A point in 2D space, in inches or pixels."""

    x: float
    y: float

    # def __str__(self) -> str:
    #     return_str = f"x: {self.x}, y: {self.y}"
    #     return return_str

# starting points
POINT_BOTTOM_GOALBOX = Point(0,-60)
POINT_TOP_GOALBOX = Point(0,60)
POINT_CENTER_FIELD = Point(0,0)

@dataclass 
class FieldState:
    timestep : float = 0
    our_bot_pos : Point = field(default_factory=lambda: POINT_BOTTOM_GOALBOX)
    their_bot_pos : Point = field(default_factory=lambda: POINT_TOP_GOALBOX)

    our_bot_bearing : float = RADIANS_NORTH
    their_bot_bearing : float = RADIANS_SOUTH

    ball_position : Point = field(default_factory=lambda: POINT_CENTER_FIELD)
  

@dataclass
class Rectangle:
    """Rectangle geometry following OpenCV's x, y, width, height convention."""

    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 100



@dataclass 
class OurRobot:
    field_pos_in : Point = field(default_factory=lambda: Point(0.0, -60))
    azimuth_rad : float = RADIANS_NORTH

    field_pos_px : Point = field(default_factory=lambda: Point(0.0, 0))

    color = CYAN
    width_in = 14

    label : str = "our bot"


@dataclass
class Detection:
    """Structure that describes the object in the frame."""

    # From the camera
    name: str = "no name"
    width_px: int = 0
    screen_center_px: Point = field(default_factory=lambda: Point(0.0, 0.0))
    area : int = 0

    # Updated with inch range and angle
    range: float = 0
    azimuth_rad: float = 0

    # Updated to a global position
    field_pos_in: Point = field(default_factory=lambda: Point(0.0, 0.0))
    field_pos_px: Point = field(default_factory=lambda: Point(0.0, 0.0))

    def __str__(self) -> str:
        return (
            f"Detection({self.name!r}, "
           #f"width_px={self.width_px}, "
           # f"area={self.area}, "
           # f"center_px={self.center_px}, "
            f"range={self.range:.2f} in, "
            f"azimuth={self.azimuth_rad:.3f} rad, "
            f"field_pos_in={round(self.field_pos_in.x), round(self.field_pos_in.y)}, "
            f"field_pos_px={round(self.field_pos_px.x), round(self.field_pos_px.y)})"
        )


