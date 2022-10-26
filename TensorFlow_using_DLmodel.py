from tensorflow.keras.applications import efficientnet_v2
import tensorflow as tf
import numpy as np
import cv2
model3 = efficientnet_v2.EfficientNetV2L()

img = cv2.imread('image/r01.jpg');
img2 = cv2.resize(img,(480,480))
#cv2.imshow("image",img2)
#cv2.waitKey(0)

img2_shape = tf.expand_dims(img2, 0)
Y3 = model3.predict(img2_shape)
Y_decoded = efficientnet_v2.decode_predictions(Y3)
label3 = Y_decoded[0][0][1]
score3 = Y_decoded[0][0][2]
print(label3)
print(score3)