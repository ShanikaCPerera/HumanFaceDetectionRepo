import os

from flask import *
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io
import base64
from tensorflow import keras
import cv2
import numpy as np


def model_prediction(image):
    



    #applying selective search
    print("selective search started")
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    print("selective search completed")
    imout = image.copy()
    n_faces = 0

    for e,result in enumerate(ssresults):
        print(e)
        if e < 700:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out= face_detection_model.predict(img)
            if out[0][0] > 0.90:
                n_faces += 1
                cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break
    
    return imout,n_faces

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
            #print(file)
            #print(type(file))
            
            #convert file from webpage to image
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            test = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
            color_image_flag = 1
            img = cv2.imdecode(test, color_image_flag)
            (h, w) = img.shape[:2]
            r = 224 / float(h)
            dim = (int(w * r), 224)
            
            



            #test to view the image
            """
            cv2.imshow("test",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            filename = secure_filename(file.filename)
            file.save(filename)
            #############
            #im1 = Image.open(file)
            #data1 = io.BytesIO()
            #im1.save(data1, "JPEG")
            #encoded_img_data1 = base64.b64encode(data1.getvalue())
            #############

            #resend image
            img_predicted,n_faces = model_prediction(img)
            #img_predicted = cv2.resize(img_predicted, dim)
            img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2RGB)
            """
            cv2.imshow("test",img_predicted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            #im = Image.open(r"C:\Users\andre\Desktop\HumanFaceDetectionRepo-main\Application\static\Images\OIP4.jpg")
            data = io.BytesIO()
            #im.save(data, "JPEG")

            img_predicted = Image.fromarray(img_predicted)

            #img_predicted = img_predicted.convert("RGB")


            img_predicted.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())







            #img = cv2.resize(img, dim)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data1 = io.BytesIO()
            im1 = Image.fromarray(img)
            im1.save(data1, "JPEG")
            encoded_img_data1 = base64.b64encode(data1.getvalue())

            # img = os.path.join(app.config['UPLOAD'], filename)
            return render_template('main.html',img_data=encoded_img_data1.decode('utf-8'),img_data1=encoded_img_data.decode('utf-8'),faceCount=n_faces)
            #return jsonify({'image_data': encoded_img_data.decode('utf-8')})

    return render_template('main.html')


if __name__ == '__main__':
    #load the model
    face_detection_model = keras.models.load_model("C:\\Users\\andre\\Desktop\\Trained Model\\exp5_rcnn_resnet50_1.h5")
    #initialize selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    app.run(debug=True, port=2000)