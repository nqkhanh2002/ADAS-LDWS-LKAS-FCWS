"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moviepy.editor import VideoFileClip
class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('Camera_Calibration_Chessboard', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)

        img = self.thresholding.forward(img)
        img = self.transform.forward(img)
        
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)
        result = mpimg.imread(output_path)
        plt.imshow(result)
        plt.show()

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)
            # Hiển thị video sau khi xử lý
        cap = cv2.VideoCapture(output_path)
        paused = False
        forward = False
        playback_speed = 1.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / (playback_speed * fps))  # Tính toán thời gian chờ dựa trên tốc độ phát lại

        while True:
            if not paused or forward:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Result Video', frame)
                key = cv2.waitKey(delay)

                if key == ord('q'):  # Phím 'q' để thoát khỏi vòng lặp
                    break
                elif key == ord('p'):  # Phím 'p' để tạm dừng hoặc tiếp tục video
                    paused = not paused
                elif key == ord('r'):  # Phím 'r' để quay lại đầu video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    paused = False
                elif key == ord('f'):  # Phím 'f' để tua nhanh video
                    forward = True
                else:
                    forward = False

        cap.release()
        cv2.destroyAllWindows()
        

def main():
    args = docopt(__doc__)
    input = args['INPUT_PATH']
    output = args['OUTPUT_PATH']

    findLaneLines = FindLaneLines()
    if args['--video']:
        findLaneLines.process_video(input, output)
    else:
        findLaneLines.process_image(input, output)
    # input = 'Test_Video/video_test_01.mp4'
    # output = 'Results/output.mp4'
    # findLaneLines = FindLaneLines()
    # findLaneLines.process_video(input, output)

if __name__ == "__main__":
    main()

    