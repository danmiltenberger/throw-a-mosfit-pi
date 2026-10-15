from dataclasses import dataclass, field
from typing import Optional
from tam_library_colors import YELLOW
import numpy as np





@dataclass
class Point:
    """A point in 2D space, in inches or pixels."""

    x: float
    y: float

    # def __str__(self) -> str:
    #     return_str = f"x: {self.x}, y: {self.y}"
    #     return return_str

  

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
    azimuth_rad : float = -np.pi/2

    field_pos_px : Point = field(default_factory=lambda: Point(0.0, 0))

    color = YELLOW
    width_in = 14


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
            f"pos_global_in={round(self.field_pos_in.x), round(self.field_pos_in.y)}, "
            f"pos_global_px={round(self.field_pos_px.x), round(self.field_pos_px.y)})"
        )