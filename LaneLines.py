import cv2
import numpy as np
import matplotlib.image as mpimg

def hist(img):
    """
    Compute the histogram of the bottom half of the input image.

    Args:
        img (np.array): Input image.

    Returns:
        np.array: Histogram of pixel intensities.
    """
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """
    Class for detecting and tracking lane lines in an image.
    """

    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread(r'Image_Resrouces\retrai.png')
        self.right_curve_img = mpimg.imread(r'Image_Resrouces\rephai.png')
        self.keep_straight_img = mpimg.imread(r'Image_Resrouces\dithang.png')

        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the windows +/- margin
        self.margin = 100
        # Minimum number of pixels found to recenter window
        self.minpix = 50

    def forward(self, img):
        """
        Detect and track lane lines in the input image.

        Args:
            img (np.array): Input image.

        Returns:
            np.array: Binary image with the detected lane lines.
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """
        Get the x and y positions of pixels within a window around the specified center.

        Args:
            center (tuple): Center of the window.
            margin (int): Half the width of the window.
            height (int): Height of the window.

        Returns:
            tuple: Arrays of x and y positions of pixels within the window.
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        """
        Extract features for lane line detection.

        Args:
            img (np.array): Input image.
        """
        self.img = img
        # Height of windows - based on nwindows and image shape
        self.window_height = np.int(img.shape[0]//self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """
        Find lane pixels using a sliding window approach.

        Args:
            img (np.array): Binary image.

        Returns:
            tuple: Lists of x and y positions of left and right lane pixels, and the output image.
        """
        assert(len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        # Create empty lists to receive left and right lane pixel coordinates
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img


    def fit_poly(self, img):
        """
        Fit polynomial curves to the detected lane pixels.

        Args:
            img (np.array): Binary image.

        Returns:
            np.array: Output image with polynomial curves plotted on it.
        """
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Visualization
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        lR, rR, pos = self.measure_curvature()

        return out_img


    def plot(self, out_img):
        """
        Plot lane lines and text overlays on the image.

        Args:
            out_img (np.array): Image with polynomial curves plotted on it.

        Returns:
            np.array: Image with lane lines, text overlays, and direction indicators.
        """
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()

        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')

        if len(self.dir) > 10:
            self.dir.pop(0)

        W = 400
        H = 430
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0, :] = [0, 0, 255]
        widget[-1, :] = [0, 0, 255]
        widget[:, 0] = [0, 0, 255]
        widget[:, -1] = [0, 0, 255]
        out_img[:H, :W] = widget

        direction = max(set(self.dir), key=self.dir.count)
        msg = "LKAS: Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        if direction == 'L':
            y, x = self.left_curve_img[:, :, 3].nonzero()
            out_img[y, x - 100 + W // 2] = self.left_curve_img[y, x, :3]
            msg = "LKAS: Left Curve Ahead"
        if direction == 'R':
            y, x = self.right_curve_img[:, :, 3].nonzero()
            out_img[y, x - 100 + W // 2] = self.right_curve_img[y, x, :3]
            msg = "LKAS: Right Curve Ahead"
        if direction == 'F':
            y, x = self.keep_straight_img[:, :, 3].nonzero()
            out_img[y, x - 100 + W // 2] = self.keep_straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(40, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(40, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)

        cv2.putText(out_img, "LDWS: Good Lane Keeping", org=(10, 350), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)

        cv2.putText(out_img, "Vehicle is {:.2f}m away from center".format(pos), org=(10, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.66, color=(255, 255, 255), thickness=2)

        return out_img


    def measure_curvature(self):
        """
        Measure the curvature of the lane lines.

        Returns:
            Tuple: Left lane curvature, right lane curvature, and position of the vehicle from the lane center.
        """
        ym = 30 / 720
        xm = 3.7 / 700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        xl = np.dot(self.left_fit, [700 ** 2, 700, 1])
        xr = np.dot(self.right_fit, [700 ** 2, 700, 1])
        pos = (1280 // 2 - (xl + xr) // 2) * xm
        return left_curveR, right_curveR, pos
