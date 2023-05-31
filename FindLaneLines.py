"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""
import os
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import tkinter as tk
def extract_label_from_calib(calib_path):
    with open(calib_path, 'r') as f:
        content = f.read().strip().split('\n')

    label_line = content[2].split(' ')[1:]  # Assuming P2 is in the third line (index 2)
    label = [float(part) for part in label_line]  # Convert each part to float

    return label



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
        
        img = self.calibration.undistort(img)
        # resize video
        out_img = np.copy(img)
        img = self.thresholding.forward(img)
        img = self.transform.forward(img)
        
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img
    def forward_image(self, img):
        # rows, cols = img.shape[:2]
        # Lấy chiều cao và chiều rộng của hình ảnh
        height, width = img.shape[:2]
        # Cắt đôi hình ảnh theo chiều dọc
        half_height = height // 2
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))  # Tạo subplot với kích thước 2x3
        axs[0, 0].imshow(img)  # Hiển thị ảnh gốc
        axs[0, 0].set_title('Original Image')

        img = self.calibration.undistort(img)
        out_img = np.copy(img)
        # img = img[half_height:, :]
        axs[0, 1].imshow(img)  # Hiển thị ảnh gốc
        axs[0, 1].set_title('Undistorted Image')
        
        
        img = self.thresholding.forward(img)
        axs[0, 2].imshow(img, cmap='gray')  # Hiển thị ảnh sau khi áp dụng thresholding
        axs[0, 2].set_title('Thresholded Image')

        img = self.transform.forward(img)
        axs[1, 0].imshow(img, cmap='gray')  # Hiển thị ảnh sau khi áp dụng perspective transformation
        axs[1, 0].set_title('Transformed Image')

        img = self.lanelines.forward(img)
        axs[1, 1].imshow(img, cmap='gray')  # Hiển thị ảnh sau khi xử lý lane lines
        axs[1, 1].set_title('Lane Lines Image')

        img = self.transform.backward(img)
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        axs[1, 2].imshow(out_img)  # Hiển thị ảnh cuối cùng sau khi plot lane lines
        axs[1, 2].set_title('Final Image with Lane Lines')

        # plt.tight_layout()  # Đảm bảo không bị trùng lấp subplot
        # plt.show()

        return out_img

    def process_image(self, input_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        return out_img
        # mpimg.imsave(output_path, out_img)
        # result = mpimg.imread(output_path)
        # plt.imshow(result)
        # plt.show()

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)
        # Hiển thị video sau khi xử lý
        # cap = cv2.VideoCapture(output_path)
        # paused = False
        # forward = False
        # playback_speed = 1.0
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # delay = int(1000 / (playback_speed * fps))  # Tính toán thời gian chờ dựa trên tốc độ phát lại

        # while True:
        #     if not paused or forward:
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
        #         cv2.imshow('Result Video', frame)
        #         key = cv2.waitKey(delay)

        #         if key == ord('q'):  # Phím 'q' để thoát khỏi vòng lặp
        #             break
        #         elif key == ord('p'):  # Phím 'p' để tạm dừng hoặc tiếp tục video
        #             paused = not paused
        #         elif key == ord('r'):  # Phím 'r' để quay lại đầu video
        #             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #             paused = False
        #         elif key == ord('f'):  # Phím 'f' để tua nhanh video
        #             forward = True
        #         else:
        #             forward = False
        # cap.release()
        # cv2.destroyAllWindows()
     
    

    def evaluate(self, dataset_path):
        true_labels = []
        predicted_labels = []
        lane_detected = 0

        calib_path = os.path.join(dataset_path, 'calib')
        image_path = os.path.join(dataset_path, 'image_2')

        for file in os.listdir(image_path):
            if file.endswith('.png') and file.startswith('um_'):
                image_file_path = os.path.join(image_path, file)
                calib_file_path = 'um' + file[2:-4] + '.txt'
                calib_file_path = os.path.join(calib_path, calib_file_path)

                

                try:
                    predicted_label = self.process_image(image_file_path)
                    predicted_labels.append(predicted_label)
                    lane_detected += 1
                except:
                    continue
                
                true_label = extract_label_from_calib(calib_file_path)
                true_labels.append(true_label)

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        # lane_detection_rate = lane_detected / len(predicted_labels)

        # return precision, recall, f1, lane_detection_rate
        return precision, recall, f1

        

# def main():
#     args = docopt(__doc__)
#     input = args['INPUT_PATH']
#     output = args['OUTPUT_PATH']

#     findLaneLines = FindLaneLines()
#     if args['--video']:
#         findLaneLines.process_video(input, output)
#     else:
#         findLaneLines.process_image(input, output)

# if __name__ == "__main__":
#     main()