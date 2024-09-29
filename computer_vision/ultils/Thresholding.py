import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    """
    Perform relative thresholding on the input image.

    Args:
        img (np.array): Input image.
        lo (float): Lower threshold value.
        hi (float): Upper threshold value.

    Returns:
        np.array: Binary image representing the relevant pixels.
    """
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    """
    Perform absolute thresholding on the input image.

    Args:
        img (np.array): Input image.
        lo (int): Lower threshold value.
        hi (int): Upper threshold value.

    Returns:
        np.array: Binary image representing the relevant pixels.
    """
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    """
    Class for extracting relevant pixels in an image.
    """

    def __init__(self):
        pass

    def forward(self, img):
        """
        Take an image and extract all relevant pixels.

        Args:
            img (np.array): Input image.

        Returns:
            np.array: Binary image representing the positions of relevant pixels.
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        v_channel = hsv[:,:,2]

        right_lane = threshold_rel(l_channel, 0.8, 1.0)
        right_lane[:,:750] = 0

        left_lane = threshold_abs(h_channel, 20, 30)
        left_lane &= threshold_rel(v_channel, 0.7, 1.0)
        left_lane[:,550:] = 0

        img2 = left_lane | right_lane

        return img2
