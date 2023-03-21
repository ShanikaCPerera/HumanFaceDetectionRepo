import os

from flask import *
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')


upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder


@app.route('/home', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
            file = request.files['imageFile']
            filename = secure_filename(file.filename)
            file.save(filename)
            #############
            im1 = Image.open(file)
            data1 = io.BytesIO()
            im1.save(data1, "JPEG")
            encoded_img_data1 = base64.b64encode(data1.getvalue())
            #############

            #resend image
            im = Image.open(r"C:\Users\nikit\Downloads\HumanFaceDetectionRepo-main\HumanFaceDetectionRepo-main\Application\OIP4.jpg")
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())

            # img = os.path.join(app.config['UPLOAD'], filename)
            return render_template('main.html',img_data=encoded_img_data.decode('utf-8'),img_data1=encoded_img_data1.decode('utf-8'),faceCount=5)
            #return jsonify({'image_data': encoded_img_data.decode('utf-8')})

    return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)