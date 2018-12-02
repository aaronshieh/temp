import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2, os

model = load_model('model_20181130.h5')

for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'test')):
    for file in files:
        if file.split('.')[-1] in ['jpg', 'png']:
            print(file)
            img = cv2.imread(os.path.join(os.getcwd(), 'test', file), 0)
            img = cv2.resize(img, (100, 100))
            img = cv2.equalizeHist(img)
            img = img/255

            print(file, "predicted:", np.argmax(model.predict(img.reshape(-1,100,100,1))))
            print(model.predict(img.reshape(-1,100,100,1)))
            print("="*100)

print("end")