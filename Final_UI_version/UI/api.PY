from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io
import base64
from tensorflow import keras
import cv2
import numpy as np
import time
from keras.layers import  Input
import Faster_RCNN
from keras.models import Model
from keras import backend as K

# Predict the results for faster RCNN
def faster_RCNN_prediction(img):
    C = Faster_RCNN.Config()
    base_weight_path = 'C:\\SoftwareEngineeringProject\\Final_UI_version\\Final_UI_version\\UI\\vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    output_weight_path = 'C:\\SoftwareEngineeringProject\\Final_UI_version\\Final_UI_version\\UI\\model_frcnn_vgg_1.hdf5'
    C.model_path = output_weight_path
    

    num_faces = 0
    C.base_net_weights = base_weight_path

    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = Faster_RCNN.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = Faster_RCNN.rpn_layer(shared_layers, num_anchors)
    
    class_mapping = {'face': 0, 'bg': 1}
    class_mapping = {v: k for k, v in class_mapping.items()}
    classifier = Faster_RCNN.classifier_layer(feature_map_input, roi_input, C.num_rois, num_classes=len(class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')


    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    # If the box classification value is less than this, we ignore this box
    bbox_threshold = 0.7

    X, ratio = Faster_RCNN.format_img(img, C)

    X = np.transpose(X, (0, 2, 3, 1))

    # get output layer Y1, Y2 from the RPN and the feature maps F
    # Y1: y_rpn_cls
    # Y2: y_rpn_regr
    [Y1, Y2, F] = model_rpn.predict(X)

    # Get bboxes by applying NMS 
    # R.shape = (300, 4)
    R = Faster_RCNN.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = Faster_RCNN.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = Faster_RCNN.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]
            num_faces += 1
            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = Faster_RCNN.get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    return img, num_faces


def non_max_suppression_RCNN(boxes, overlapThresh):
    """
    Remove overlapping bounding boxes that refer to the same object
    in the input aray of bounding boxes.
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


#prediction for RCNN
def model_prediction(image):
    #applying selective search
    start_time_ss = time.perf_counter()
    print("selective search started")
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    print("selective search completed")
    finish_time_ss = time.perf_counter()
    total_time_for_ss = int(finish_time_ss - start_time_ss)
    imout = image.copy()
    imout2 = image.copy()
    n_faces = 0
    rects = []
    start_time_model = time.perf_counter()
    for e,result in enumerate(ssresults):
        print(e)
        if e < 2000:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out= RCNN_Model.predict(img)
            if out[0][0] > 0.90:

                rects.append([x, y, x+w, y+h])
        else:
            break
    finish_time_model = time.perf_counter()
    total_time_for_model = int(finish_time_model - start_time_model)

    rects_ = np.array(rects)
    pick = non_max_suppression_RCNN(rects_, overlapThresh=0.40)

    for (xA, yA, xB, yB) in pick:
        n_faces += 1
        cv2.rectangle(imout2, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    return imout2,n_faces,total_time_for_ss,total_time_for_model

app = Flask(__name__)


#Default route when page will be loaded
@app.route('/')
def home():
    return render_template('main.html')


upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder


#route after detect button is clicked
@app.route('/home', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['imageFile']  #retrieving image uploaded from the UI
        var = request.form.get('selectId') # retrieving model type selected from UI
        print(str(var))
        #convert file from webpage to image
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        test = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        original_image = cv2.imdecode(test, color_image_flag)
        original_image_1 = cv2.imdecode(test, color_image_flag)
        (h, w) = original_image.shape[:2]
        r = 224 / float(h)
        dim = (int(w * r), 224)

        filename = secure_filename(file.filename)
        file.save(filename)
        total_time_for_ss = 0
        total_time_for_model = 0
        if(var=='RCNN'): # RCNN if block to predict
            img_predicted,n_faces,total_time_for_ss,total_time_for_model = model_prediction(original_image)

        if (var == 'F-RCNN'): # F-RCNN if block to predict
            img_predicted,n_faces = faster_RCNN_prediction(original_image)

        img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2RGB)

        data = io.BytesIO()

        img_predicted = Image.fromarray(img_predicted)

        img_predicted.save(data, "JPEG")
        encoded_img_data = base64.b64encode(data.getvalue())

        original_image_1 = cv2.cvtColor(original_image_1, cv2.COLOR_BGR2RGB)
        data1 = io.BytesIO()
        im1 = Image.fromarray(original_image_1)
        im1.save(data1, "JPEG")
        encoded_img_data1 = base64.b64encode(data1.getvalue())

        #return to main.html with  img_data: original image
        #img_data1: processed image
        #faceCount: count of faces in image
        #model_used: type of model either RCNN/FRCNN
        #ss_time: time required to finish selective search
        #model_time: time required to process the model
        return render_template('main.html',img_data=encoded_img_data1.decode('utf-8'),img_data1=encoded_img_data.decode('utf-8'),faceCount=n_faces,model_used = var, ss_time = total_time_for_ss,model_time = total_time_for_model)

    #return to main.html
    return render_template('main.html')


#main method will load first when server will be up
if __name__ == '__main__':
    #load the model
    RCNN_Model = keras.models.load_model('C:\\SoftwareEngineeringProject\\Final_UI_version\\Final_UI_version\\UI\\exp4_rcnn_imagenet_kfcv.h5')

    #initialize selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    app.run(debug=True, port=2000)