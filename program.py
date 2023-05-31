from flask import *  

import pandas as pd
import numpy as np
import os

import pickle
import cv2
import glob
from PIL import Image
from random import randint

from FindLaneLines import *

IMAGE_FOLDER = 'static/'
PROCESSED_FOLDER = 'static/Results/Image/'

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


@app.route('/')  
def upload():
	return render_template("homepage.html")  
 
@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
		full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
		
		# Kiểm tra định dạng tệp tin
		is_image = False
		is_video = False
		try:
			image = Image.open(full_filename)
			is_image = True
		except Exception:
			try:
				video_clip = VideoFileClip(full_filename)
				is_video = True
			except Exception:
				pass
				return 'Invalid file format.'
		
		# Xử lý tùy thuộc vào định dạng tệp tin
		method_cv = FindLaneLines()
		if is_image:
			output_image_after_detecting = method_cv.process_image(full_filename)
			i = randint(1, 1000000)
			char = str(i)
			hls_name = 'output_' + f.filename + '_' + char + '.jpg'
			mpimg.imsave('static/Results/Image/'+ hls_name, output_image_after_detecting)
			full_filename_processed = os.path.join(app.config['PROCESSED_FOLDER'], hls_name)
			
			final_text = ''
			return render_template("result.html",type_file = 'image', name=final_text, img_in=full_filename, img=full_filename_processed)
		elif is_video:
			# Xử lý video
			method_cv = FindLaneLines()
			i = randint(1, 1000000)
			char = str(i)
			hls_name = 'output_' + f.filename + '_' + char + '.mp4'
			output_path = 'static/Results/Video/'+ hls_name			
			method_cv.process_video(full_filename, output_path)
			final_text = ''
			return render_template("result.html", type_file = 'video', name=final_text, video_in=full_filename, video=output_path)

		else:
			return 'Invalid file format.'

# @app.route('/info', methods = ['POST'])
# def info():
# 	if request.method == 'POST':
# 		return render_template("info.html")

if __name__ == '__main__':  
	app.run(host="127.0.0.1",port=8080,debug=True)