from flask import Flask,request,render_template,redirect
import os
import cv2
import imutils
import numpy


app = Flask(__name__,template_folder=r'C:\Users\nikit\Downloads\HumanFaceDetectionRepo-main\HumanFaceDetectionRepo-main\PythonAPI\templates')
#app = Flask(__name__,template_folder=r'\templates')

cascPath = '/'
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


app.config["IMAGE_UPLOADS"] = r"C:\Users\nikit\Downloads\HumanFaceDetectionRepo-main\HumanFaceDetectionRepo-main\PythonAPI\static\Images"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename


@app.route('/home',methods = ["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']
		cv2.imshow(image)
		#read image file string data
		filestr = request.files['file'].read()
		#convert string data to numpy array
		file_bytes = numpy.fromstring(filestr, numpy.uint8)
		# convert numpy array to image
		img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Detect faces in the image
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=2,
			minSize=(5, 5)
			#flags = cv2.CV_HAAR_SCALE_IMAGE
		)
		i = 1
		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
			cv2.putText(img, 'face '+str(i), (x-10, y-10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			i += 1
		filename = secure_filename(image.filename)
		cv2.imwrite(r"C:\Users\nikit\Downloads\HumanFaceDetectionRepo-main\HumanFaceDetectionRepo-main\PythonAPI\static\Images" + filename ,img)


		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)


		#filename = secure_filename(image.filename)

		#basedir = os.path.abspath(os.path.dirname(__file__))
		#image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))




		return render_template("main.html",filename=filename)
	


	return render_template('main.html')

"""
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static',filename = "Images/" + filename), code=301)
"""

app.run(debug=True,port=2000)