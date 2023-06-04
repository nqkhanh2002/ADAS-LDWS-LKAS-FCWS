import cv2
import numpy as np

class PerspectiveTransformation:
    """
    Class for performing perspective transformation on images.
    """

    def __init__(self):
        # Define source and destination points for perspective transformation
        self.src = np.float32([(550, 460),     # top-left
                               (150, 720),     # bottom-left
                               (1200, 720),    # bottom-right
                               (770, 460)])    # top-right
        self.dst = np.float32([(100, 0),
                               (100, 720),
                               (1100, 720),
                               (1100, 0)])
        # Calculate the perspective transformation matrix and its inverse
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """
        Apply forward perspective transformation to the input image.
        """
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """
        Apply backward perspective transformation to the input image.
        """
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)
