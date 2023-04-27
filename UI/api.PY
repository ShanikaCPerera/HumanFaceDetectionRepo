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



def non_max_suppression_fast(boxes, overlapThresh):
    """
    Remove overlapping bounding boxes that refer to the same object
    in the input aray of bounding boxes.
    Malisiewicz et al.
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        # add the last index in the indexes list
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def model_prediction(image):
    



    #applying selective search
    print("selective search started")
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    print("selective search completed")
    imout = image.copy()
    imout2 = image.copy()
    n_faces = 0
    rects = []
    for e,result in enumerate(ssresults):
        print(e)
        if e < 2000:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out= face_detection_model.predict(img)
            if out[0][0] > 0.90:

                rects.append([x, y, x+w, y+h])
        else:
            break

    rects_ = np.array(rects)
    pick = non_max_suppression_fast(rects_, overlapThresh=0.40)

    for (xA, yA, xB, yB) in pick:
        n_faces += 1
        cv2.rectangle(imout2, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    return imout2,n_faces

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
  

            filename = secure_filename(file.filename)
            file.save(filename)


            #resend image
            img_predicted,n_faces = model_prediction(img)

            img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2RGB)

            data = io.BytesIO()


            img_predicted = Image.fromarray(img_predicted)



            img_predicted.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data1 = io.BytesIO()
            im1 = Image.fromarray(img)
            im1.save(data1, "JPEG")
            encoded_img_data1 = base64.b64encode(data1.getvalue())


            return render_template('main.html',img_data=encoded_img_data1.decode('utf-8'),img_data1=encoded_img_data.decode('utf-8'),faceCount=n_faces)


    return render_template('main.html')


if __name__ == '__main__':
    #load the model
    face_detection_model = keras.models.load_model("C:\\Users\\andre\\Desktop\\Trained Model\\exp5_rcnn_resnet50_2.h5")
    #initialize selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    app.run(debug=True, port=2000)