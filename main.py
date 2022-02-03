import numpy as np
import tensorflow as tf
import cv2
from glob import glob
import os

os.makedirs('result', exist_ok=True)

THRESHOLD = 0.3

# Load the model
interpreter = tf.lite.Interpreter(model_path='4leaf_detector.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load images
img_list = glob('detect4clover20200621/img/*.jpg')

for img_path in img_list:
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Preprocessing
    input_data = cv2.resize(img, (320, 320))
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_data, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    locations = interpreter.get_tensor(output_details[0]['index'])
    # classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    # Postprocessing
    for location, score in zip(locations[0], scores[0]):
        if score < THRESHOLD:
            continue

        if location[0] > 1. or location[1] > 1. or location[2] > 1. or location[3] > 1.:
            continue

        y1, x1, y2, x2 = (location * [h, w, h, w]).astype(int)

        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3)

    cv2.imshow('result', img)
    cv2.imwrite('result/%s' % (os.path.basename(img_path),), img)
    if cv2.waitKey(1) == ord('q'):
        break