from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('Number.model100')

image_num = 1
while os.path.isfile(f"numbers/num{image_num}.png"):
    try:
        img = cv2.imread(f"numbers/num{image_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("Det är förmodligen: " + str(np.argmax(prediction)))
        plt.imshow(img[0],cmap=plt.cm.binary)
    except:
        print("Error")
    finally:
        image_num += 1


