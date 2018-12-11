from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import numpy as np
import argparse
from imutils import paths
import imutils
import pickle
import cv2
import os
import pymongo

opencv_face_recognition = "opencv-face-recognition"

print("[INFO] loading face detector...")
protoPath = os.path.join(opencv_face_recognition, "face_detection_model", "deploy.prototxt")
modelPath = os.path.join(opencv_face_recognition, "face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face embedder model...")
embedder = cv2.dnn.readNetFromTorch(os.path.join(opencv_face_recognition, "openface_nn4.small2.v1.t7"))

def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]
    print(box)

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

def base64_to_nparray(base64_str):
    base64_str = base64_str.replace('data:image/png;base64,', '')
    imgdata = base64.b64decode(base64_str)
    nparr = np.fromstring(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def extract_face(image, square=False):
    # path = 'opencv-face-recognition'
    # CONFIDENCE = 0.5

    # print("[INFO] loading face detector...")
    # protoPath = os.path.sep.join([path, 'face_detection_model', "deploy.prototxt"])
    # modelPath = os.path.sep.join([path, 'face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
    # detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    (h, w) = image.shape[:2]
    
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if square:
                box = get_square_box(box.astype("int"))
                (startX, startY, endX, endY) = box

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
                
    return face, startX, startY, endX, endY

@csrf_exempt
def identify_face(request):
    if request.method == 'POST':
        imgString = request.POST['image']
        img = base64_to_nparray(imgString)

        path = 'opencv-face-recognition'
        imagePath = 'cameraCapture.png'
        CONFIDENCE = 0.5

        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        embedding_model = 'openface_nn4.small2.v1.t7'
        embedding_modelPath = os.path.sep.join([path, embedding_model])
        embedder = cv2.dnn.readNetFromTorch(embedding_modelPath)

        # load the actual face recognition model along with the label encoder
        print("[INFO] loading face recognition model and label encoder...")
        recognizerPath = os.path.sep.join([path, 'output', 'recognizer.pickle'])
        recognizer = pickle.loads(open(recognizerPath, "rb").read())
        lePath = os.path.sep.join([path, 'output', 'le.pickle'])
        le = pickle.loads(open(lePath, "rb").read())

        try:
            face, startX, startY, endX, endY = extract_face(img)

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            response = {
                "name": name.title(),
                "probability": proba,
                "startX": str(startX),
                "startY": str(startY),
                "endX": str(endX),
                "endY": str(endY),
                "result": "success"
            }

        except:
            response = {
                "result": "error"
            }

        return JsonResponse(response)

@csrf_exempt
def identify_emotion(request):
    if request.method == 'POST':
        imgString = request.POST['image']
        img = base64_to_nparray(imgString)

        path = 'emotion-recognition'

        # TODO: find more efficient way to load model
        model = load_model(os.path.sep.join([path, 'model_20181130.h5']))

        path_ = os.path.sep.join([path, 'test'])

        try:
            face, startX, startY, endX, endY = extract_face(img, square=True)
            
            # test_file = 'test.png'
            # test_path = os.path.join(path, 'test', test_file)
            # cv2.imwrite(test_path, face)

            # img = cv2.imread(test_path, 0)
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            # cv2.imshow("g", face)
            face = cv2.resize(face, (100, 100))
            # cv2.imshow("r", face)
            face = cv2.equalizeHist(face)
            # cv2.imshow("e", face)
            face = face/255

            # cv2.waitKey()
            # cv2.destroyAllWindows()

            proba = model.predict(face.reshape(-1,100,100,1))
            pred = np.argmax(model.predict(face.reshape(-1,100,100,1)))

            emotions = ['anger', 'happy', 'neutral', 'sad', 'surprise']

            print("predicted:", emotions[pred])
            print(proba)

            response = {
                "emotion": emotions[pred],
                "startX": str(startX),
                "startY": str(startY),
                "endX": str(endX),
                "endY": str(endY)
            }

        except:
            response = {
                "emotion": "error"
            }

        finally:
            tf.keras.backend.clear_session()

        return JsonResponse(response)

def get_face_embedding_vector(imagePaths):
    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                return vec.flatten()

def calc_face_embedding_vector(face):
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
        (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    return vec.flatten()

def add_to_embeddings(face_vector, name):
    client = pymongo.MongoClient()
    db = client.embeddings
    myMembers = db.members

    print(type(face_vector))

    member = {
        'name': name,
        'vector': face_vector.tolist()
    }
    
    myMembers.insert_one(member)

    print(f'added {name} to embeddings')

@csrf_exempt
def new_member(request):
    if request.method == 'POST':
        account = request.POST['name']
        photo = request.POST['image']

        client = pymongo.MongoClient()
        db = client.embeddings
        myMembers = db.members

        account_ = ''
        for name in myMembers.find({'name':account}):
            account_ = name
        
        if account_ != '':
            return JsonResponse({'result':'already in db'})
        else:
            try:
                img = base64_to_nparray(photo)
                face, startX, startY, endX, endY = extract_face(img)
                face_vector = calc_face_embedding_vector(face)
                add_to_embeddings(face_vector, account)
                return JsonResponse({'result':'success'})
            except:
                return JsonResponse({'result':'error'})

@csrf_exempt
def recognize_face(request):
    if request.method == 'POST':
        imgString = request.POST['image']
        img = base64_to_nparray(imgString)

        client = pymongo.MongoClient()
        db = client.embeddings
        myMembers = db.members

        try:
            face, startX, startY, endX, endY = extract_face(img)
            test_vector = calc_face_embedding_vector(face)

            l2_dist = 100
            name = 'unknown'
            for member in myMembers.find():
                temp_l2_dist = np.sqrt(np.sum(np.square(test_vector - np.array(member['vector']))))
                print(temp_l2_dist)
                if temp_l2_dist < l2_dist and temp_l2_dist <= 0.80:
                    l2_dist = temp_l2_dist
                    name = member['name']
            print("min dist.", l2_dist)

            if name == 'unknown':
                response = {
                    "result": "error"
                }
            else:
                response = {
                    "name": name,
                    "startX": str(startX),
                    "startY": str(startY),
                    "endX": str(endX),
                    "endY": str(endY),
                    "result": "success"
                }
        except:
            response = {
                "result": "error"
            }

        return JsonResponse(response)

def get_members(request):
    client = pymongo.MongoClient()
    db = client.embeddings
    myMembers = db.members

    members = {
        'names': []
    }
    for member in myMembers.find():
        members['names'].append(member['name'])

    return JsonResponse(members)

def del_members(request):
    client = pymongo.MongoClient()
    db = client.embeddings
    myMembers = db.members
    
    return JsonResponse({"result":myMembers.delete_many({}).deleted_count})