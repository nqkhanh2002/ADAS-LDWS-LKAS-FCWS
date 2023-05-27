import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
class CameraCalibration():
    """ Class that calibrate camera using chessboard images.

    Attributes:
        mtx (np.array): Camera matrix 
        dist (np.array): Distortion coefficients
    """
    def __init__(self, image_dir, nx, ny, debug=False):
        """ Init CameraCalibration.

        Parameters:
            image_dir (str): path to folder contains chessboard images
            nx (int): width of chessboard (number of squares)
            ny (int): height of chessboard (number of squares)
        """
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []
        
        # Coordinates of chessboard's corners in 3D
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # Go through all chessboard images
        for f in fnames:
            img = mpimg.imread(f)

            # Convert to grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(img, (nx, ny))
            if ret:
                
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (7,6), corners, ret)

        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        """ Return undistort image.

        Parameters:
            img (np.array): Input image

        Returns:
            Image (np.array): Undistorted image
        """
        # Convert to grayscale image
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
