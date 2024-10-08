import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
class CameraCalibration():
    def __init__(self, image_dir, nx, ny, debug=False):
        fnames = glob.glob(f"{image_dir}/*")
        # print("Fnames: ", fnames)
        objpoints = []
        imgpoints = []
        
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        shape = None
        
        # Go through all chessboard images
        for f in fnames:
            img = mpimg.imread(f)
            if shape is None:
                shape = (img.shape[1], img.shape[0])
            
            # Convert to grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
            
            if debug:
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.imshow(img)
                plt.show()

        if not shape:
            raise ValueError("No valid images found in the specified directory")

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
