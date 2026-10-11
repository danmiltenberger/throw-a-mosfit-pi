import numpy as np
import cv2



BLACK = (0,0,0)
WHITE = (255,255,255)
VERY_DARK_GRAY = (255,255,255)

FILLED = -1


def get_rect_points(center_x, center_y, width,height):
    left  = center_x - width/2
    right = center_x + width/2

    top = center_y + height/2
    bottom = center_y - height/2

    point1 = (left, top)
    point2 = (right, bottom)

    return point1, point2
    
    


def draw_empty_field():

    Window_in_x = 8*12 + 2*(2+2)
    Window_in_y = 12*12 + 2*(2+2)

    inches_to_pixels = 10
    Window_px_x = Window_in_x * inches_to_pixels
    Window_px_y = Window_in_y * inches_to_pixels

    window = np.zeros((Window_px_x, Window_px_y))



    cv2.rectangle(window, (0,0), (Window_px_x, Window_px_y), WHITE, thickness=FILLED)

    cv2.imshow("field", window)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    draw_empty_field()

if __name__ == "__main__":
    main()